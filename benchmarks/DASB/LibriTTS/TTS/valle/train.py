#!/usr/bin/env/python3
"""Recipe for training a Text-to-Speech system based on tokenized audio

Inspired by WhisperSpeech
https://github.com/collabora/WhisperSpeech

However, this is not an implementation of WhisperSpeech, but rather
a radical simplification of it that uses only an acoustic model


Authors
 * Artem Ploujnikov 2024
"""


import logging
import speechbrain as sb
import torch
import sys
import shutil
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import (
    clean_padding_,
    length_to_mask,
    write_audio,
)
from speechbrain.utils.data_utils import pad_right_to
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import batch_pad_right
from functools import partial
import re
import string

base_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(base_dir)

from evaluation import SpeechEvaluationMetricStats  # noqa: E402

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_COUNT = 1


# Brain class for speech recognition training
class VALLEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super().__init__(
            modules, opt_class, hparams, run_opts, checkpointer,
        )
        self.evaluation_metric = SpeechEvaluationMetricStats(
            self.hparams, self.device
        )

    def create_waveform(self, audio, length):
        """Creates a waveform from a discrete or continuous audio
        representation

        Arguments
        ---------
        audio : torch.Tensor
            An audio tensor (Batch x Length x Heads or Batch x Length x Heads x Features)
        lengths : torch.Tensor
            A 1-D tensor

                    Returns
        -------
        wav : torch.Tensor
        """
        self.modules.tokenizer.device = self.device
        if hasattr(self.modules.tokenizer, "codec_vocoder"):
            self.modules.tokenizer.codec_vocoder.to(self.device)
            self.modules.tokenizer.codec_vocoder.device = self.device
        wav = self.modules.tokenizer.tokens_to_sig(audio)
        clean_padding_(wav, length)
        wav = wav.to(self.device)
        return wav

    def compute_forward(self, batch, stage):
        """Runs all the computation of the Tokotron TTS

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            TTS predictions
        """
        batch = batch.to(self.device)
        prompt, prompt_length = batch.prompt
        batch_size, prompt_max_len, num_tracks = prompt.shape
        nar_track = torch.randint(
            1, num_tracks, (batch_size,), device=self.device
        )
        logits_ar, logits_nar = self.modules.model(
            dec_seq=batch.prompt.data,
            dec_seq_lengths=batch.prompt.lengths,
            prefix_len=batch.prefix_length / prompt_max_len,
            nar_level_idx=nar_track,
        )
        return logits_ar, logits_nar, nar_track

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)

        logits_ar, logits_nar, nar_track = predictions
        prompt, prompt_length = batch.prompt
        prefix_length = batch.prefix_length

        logits_ar_sm = self.hparams.log_softmax(logits_ar)
        logits_nar_sm = self.hparams.log_softmax(logits_nar)
        batch_size, max_len, _ = prompt.shape
        targets_ar = prompt[:, 1:, 0]
        batch_idx = torch.arange(batch_size, device=prompt.device)
        targets_nar = prompt[batch_idx, 1:, nar_track]
        prompt_max_len = prompt.size(1)
        length_mask = length_to_mask(
            prompt_length * prompt_max_len, prompt_max_len
        )
        prefix_mask = length_to_mask(
            prefix_length, prompt_max_len
        ).logical_not()
        mask = (length_mask * prefix_mask)[:, 1:]

        loss_ar = self.hparams.compute_cost(
            log_probabilities=logits_ar_sm, targets=targets_ar, mask=mask
        )
        self.loss_metric_ar.append(
            ids=batch.uttid,
            log_probabilities=logits_ar_sm,
            targets=targets_ar,
            mask=mask,
            reduction="batch",
        )
        loss_nar = self.hparams.compute_cost(
            log_probabilities=logits_nar_sm, targets=targets_nar, mask=mask,
        )
        self.loss_metric_nar.append(
            ids=batch.uttid,
            log_probabilities=logits_nar_sm,
            targets=targets_nar,
            mask=mask,
            reduction="batch",
        )
        loss = 0.5 * (loss_ar + loss_nar)
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        self.offsets = get_offsets(
            self.hparams.vocab_size, self.hparams.audio_tokens_per_step,
        )[None, None, :].to(self.device)

        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost, batch_eval=True,
        )
        self.loss_metric_ar = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.compute_cost, batch_eval=True,
        )
        self.loss_metric_nar = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.compute_cost, batch_eval=True,
        )

        # TOOO: Reestablish evaluation
        self.is_evaluating = False
        if stage == sb.Stage.VALID:
            if self.is_eval_epoch(epoch):
                self.evaluation_metric.on_evaluation_start()
                self.is_evaluating = True
            else:
                logger.info("No evaluation on epoch %d", epoch)
        elif stage == sb.Stage.TEST:
            self.evaluation_metric.on_evaluation_start()
            self.is_evaluating = True

    def is_eval_epoch(self, epoch):
        """Determines whether or not evaluation should be performed
        in the specieied epoch

        Arguments
        ---------
        epoch : int
            The epoch number. If omitted, the epoch number from the
            epoch counter will be used

        Returns
        -------
        eval_epoch : bool
            True if evaluation should be run in this epoch, false
            otherwise"""
        if epoch is None:
            epoch = self.hparams.epoch_counter.current
        return epoch % self.hparams.eval_interval == 0

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit/compiled modules
        # cannot be pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None and not getattr(
            self, "_ckpt_recovered", False
        ):
            self.checkpointer.recover_if_possible()
            self._ckpt_recovered = True

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        if self.is_evaluating:
            with torch.no_grad():
                audio_tokens, audio_length = self.inference(batch)
                if self.hparams.flip_layers:
                    audio_tokens = audio_tokens.flip(2)
                wav = self.create_waveform(audio_tokens, audio_length)
                wav = wav.squeeze(1)
                self.save_samples(
                    batch=batch, wav=wav, length=audio_length, stage=stage
                )
                self.evaluation_metric.append(
                    ids=batch.uttid,
                    wav=wav,
                    text=batch.label_norm_eval,
                    length=audio_length,
                    wav_ref=batch.sig.data,
                    length_ref=batch.sig.lengths,
                )
        return loss.detach().cpu()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        loss_stats = self.loss_metric.summarize(flat=True)
        stage_stats = {"loss": stage_loss, **loss_stats}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # End evaluation and report stats
        if stage != sb.Stage.TRAIN and self.is_evaluating:
            self.evaluation_metric.on_evaluation_end()
            self.save_eval(stage)
            eval_summary = self.evaluation_metric.summarize()
            eval_summary_stats = {
                key: eval_summary.get(value)
                for key, value in self.hparams.eval_summary_log.items()
            }
            stage_stats.update(eval_summary_stats)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            if self.hparams.lr_annealing_mode == "epoch":
                _, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            lr = self.optimizer.param_groups[0]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

    def inference(self, batch):
        """Runs TTS inference

        Arguments
        ---------
        batch : PaddedBatch
            A batch

        Returns
        -------
        audio : torch.Tensor
            A padded tensor of audio
        audio_length : torch.Tensor
            Relative lengths
        """
        prefix, prefix_length = batch.prefix
        # NOTE: ESPNET VALL-E does not support batched inference
        prefix_items = undo_padding_tensor(prefix.int(), prefix_length)
        inference_results = [
            self.modules.model.inference(
                prefix=prefix_item.unsqueeze(0), opts=self._get_inference_opts()
            )
            for prefix_item in prefix_items
        ]
        inferred_tokens = [
            result[0][0]
            if result[0]
            else torch.zeros(
                1000, self.hparams.audio_tokens_per_step, device=self.device
            )
            for result in inference_results
        ]
        audio, audio_length = batch_pad_right(inferred_tokens)
        audio_length = audio_length.to(self.device)
        audio = (audio - hparams["audio_token_shift"] - self.offsets).clip(0)
        return audio, audio_length

    def _get_inference_opts(self):
        idx = torch.arange(self.hparams.model_vocab_size, device=self.device)[
            None, :
        ]
        tracks = torch.arange(
            self.hparams.audio_tokens_per_step, device=self.device
        )[:, None]
        track_start = (
            self.hparams.text_num_tokens
            + self.hparams.special_num_tokens
            + tracks * self.hparams.vocab_size
        )
        if self.hparams.flip_layers:
            track_start = track_start.flip(0)
        track_end = track_start + self.hparams.vocab_size
        mask = (
            ((idx >= track_start) & (idx < track_end))
            | (idx == self.hparams.eos_index)
        ).logical_not()
        return self.hparams.inference_opts(
            masks={self.hparams.bos_index: mask}, device=self.device,
        )

    def save_samples(self, batch, wav, length, stage):
        output_folder = self._get_eval_output_folder(stage)
        samples = undo_padding_tensor(wav, length)
        for uttid, sample in zip(batch.uttid, samples):
            file_name = output_folder / f"pred_{uttid}.wav"
            write_audio(file_name, sample.cpu(), self.hparams.model_sample_rate)

    def save_eval(self, stage):
        """Saves evaluation results

        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        """
        output_folder = self._get_eval_output_folder(stage)
        for src_file_name in self.evaluation_metric.files:
            dest_file_name = output_folder / src_file_name.name
            shutil.copyfile(src_file_name, dest_file_name)
        self.evaluation_metric.clear()

    def _get_eval_output_folder(self, stage):
        epoch = self.hparams.epoch_counter.current
        output_folder = (
            Path(self.hparams.output_folder) / "eval" / stage.name.lower()
        )
        if epoch is not None:
            output_folder = output_folder / str(epoch)
        output_folder.mkdir(exist_ok=True, parents=True)
        return output_folder

    def fit_batch(self, batch):
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss


INPUT_FEATURE_MAP = {"text": "label_norm", "phonemes": "phn"}


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    silence_token : dict
        the token used for silence
    """

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    label_encoder = hparams["label_encoder"]
    input_feature = INPUT_FEATURE_MAP[hparams["input"]]
    offsets = get_offsets(
        hparams["vocab_size"], hparams["audio_tokens_per_step"]
    ).unsqueeze(0)
    if hparams["flip_layers"]:
        offsets = offsets.flip(-1)

    tokens_loader = hparams.get("tokens_loader")
    spk_prompt_length = hparams["spk_prompt_length"]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_norm", "label_norm_eval")
    def text_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        label_norm = label.upper()
        yield label_norm
        label_norm_eval = RE_PUNCTUATION.sub("", label_norm)
        yield label_norm_eval

    @sb.utils.data_pipeline.takes(input_feature)
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label_encoder.encode_sequence_torch(label)

    def spk_prompt(uttid, spk_sample):
        # Sample a speaker-matched embedding
        selected_uttid = spk_sample[uttid]
        audio = tokens_loader.tokens_by_uttid(
            selected_uttid, num_codebooks=hparams["audio_tokens_per_step"]
        )
        if audio.size(0) > spk_prompt_length:
            offset = torch.randint(0, audio.size(0), (1,)).item()
        else:
            offset = 0
        # Retrieve the embedding value from the dataset
        audio_spk_prompt, _ = pad_right_to(
            audio[offset : offset + spk_prompt_length],
            (spk_prompt_length, audio.size(1)),
        )
        return audio_spk_prompt

    @sb.utils.data_pipeline.takes("uttid", "tokens", "spk_prompt")
    @sb.utils.data_pipeline.provides(
        "audio", "prefix", "prompt", "prefix_length", "length"
    )
    def prompt_pipeline(id, tokens, spk_prompt):
        audio = tokens_loader.tokens_by_uttid(
            id, num_codebooks=hparams["audio_tokens_per_step"]
        )
        if hparams["flip_layers"]:
            audio = audio.flip(-1)
        yield audio
        num_tracks = audio.size(1)
        prefix = torch.cat(
            [
                torch.ones(1, num_tracks) * hparams["bos_index"],
                tokens.unsqueeze(-1).expand(len(tokens), num_tracks),
                torch.ones(1, num_tracks) * hparams["eot_index"],
                spk_prompt + hparams["audio_token_shift"] + offsets,
                torch.ones(1, num_tracks) * hparams["eop_index"],
            ]
        )
        yield prefix
        prompt = torch.cat(
            [
                prefix,
                torch.ones(1, num_tracks) * hparams["bos_index"],
                audio + hparams["audio_token_shift"] + offsets,
                torch.ones(1, num_tracks) * hparams["eos_index"],
            ]
        ).int()
        yield prompt
        yield len(prefix)
        yield len(prompt)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sig_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    dynamic_items = [text_pipeline, tokens_pipeline, sig_pipeline]

    init_sequence_encoder(hparams)
    use_spk_emb = hparams.get("use_spk_emb", False)
    prepared_features = ["audio_tokens"]
    output_keys = [
        "uttid",
        "tokens",
        "label_norm",
        "audio",
        "prompt",
        "prefix_length",
        "length",
    ]
    if use_spk_emb:
        prepared_features.append("spk_emb")
        output_keys.append("spk_emb")

    resample_fn = {}
    for dataset in data_info:
        dataset_dynamic_items = list(dynamic_items)
        dataset_output_keys = list(output_keys)
        if dataset != "train":
            dataset_output_keys += ["sig", "label_norm_eval", "prefix"]
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dataset_dynamic_items,
            output_keys=dataset_output_keys,
        )
        spk_idx, spk_samplers = group_by_speaker(dynamic_dataset, hparams)
        spk_sample = {}
        spk_prompt_pipeline = partial(spk_prompt, spk_sample=spk_sample,)
        dynamic_dataset.add_dynamic_item(
            func=spk_prompt_pipeline, takes=["uttid"], provides=["spk_prompt"],
        )
        dynamic_dataset.add_dynamic_item(prompt_pipeline)
        resample_fn[dataset] = partial(
            resample_spk,
            spk_idx=spk_idx,
            sample=spk_sample,
            dataset=dynamic_dataset,
            spk_samplers=spk_samplers,
        )
        resample_fn[dataset](epoch=0)

        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        if not hparams["overfit_test"]:
            hparams["train_dataloader_opts"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets


def get_offsets(vocab_size, tracks):
    """Adds offsets to each track to treat the tokens as distinct

    Arguments
    ---------
    vocab_size : int
        The vocabulary size, for each track
    tracks : int
        The number of tracks
    """
    return torch.arange(tracks) * vocab_size


def group_by_speaker(dataset, hparams):
    """Groups utterance IDs in a dataset by speaker, for selection. The selection
    is stable based on the seed - calling this method multiple times will always
    result in the same order

    Arguments
    ---------
    dataset : torch.Tensor
        the dataset from which to select items
    hparams : dict
        hyperparameters

    Returns
    -------
    spk_idx : dict
        a str -> str with a list of utterance IDs
        for every speaker
    spk_samplers : dict
        a reproducible sampler for every speaker
    spk_samplers_it : dict
        an iterator for each sampler
    """
    spk_uttid = {}
    spk_samplers = {}
    speakers = []
    generator = torch.Generator()
    generator.manual_seed(hparams["seed"])

    # Group by speaker
    with dataset.output_keys_as(["spk_id", "uttid"]):
        for idx, item in enumerate(dataset):
            spk_id = item["spk_id"]
            if spk_id not in spk_uttid:
                spk_uttid[spk_id] = []
            spk_uttid[spk_id].append(item["uttid"])
            speakers.append(spk_id)

    # Create a reproducible sampler
    for spk_id in speakers:
        sampler = hparams["spk_sampler"](data_source=spk_uttid[spk_id])
        spk_samplers[spk_id] = sampler

    return spk_uttid, spk_samplers


def resample_spk(sample, spk_idx, spk_samplers, dataset, epoch):
    """Selects new samples

    Arguments
    ---------
    spk_idx : dict
        Data item indexes grouped by speaker
    spk_samplers : dict
        A sampler for each speaker
    spk_samplers_it : dict
        An iterator for each speaker
    epoch : int
        The epoch number

    Returns
    -------
    sample : dict
        a dictionary with uttids as keys and matching
        indexes as values
    """
    if epoch is None:
        epoch = 0
    spk_samplers_it = {}
    for spk_id, sampler in spk_samplers.items():
        sampler.set_epoch(epoch)
        spk_samplers_it[spk_id] = iter(sampler)
    with dataset.output_keys_as(["uttid", "spk_id"]):
        for item in dataset:
            spk_item_idx = next(spk_samplers_it[item["spk_id"]])
            dataset_item_idx = spk_idx[item["spk_id"]][spk_item_idx]
            sample[item["uttid"]] = dataset_item_idx


def init_sequence_encoder(hparams):
    """Initialize a sequence encoder

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    prefix: str
        the prefix to be prepended to hyperparameter keys, per the naming
        convention

        {prefix}_label_encoder: the hparams key for the label encoder
        {prefix}_list_file:  the hparams key for the list file

    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder instance"""
    encoder = hparams["label_encoder"]
    token_list_file_name = hparams["token_list_file"]
    tokens = read_token_list(token_list_file_name)
    encoder.add_unk()
    encoder.update_from_iterable(tokens, sequence_input=False)
    encoder.expect_len(len(tokens) + SPECIAL_TOKEN_COUNT)
    return encoder


def read_token_list(file_name):
    """Reads a simple text file with tokens (e.g. characters or phonemes) listed
    one per line

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    result: list
        a list of tokens
    """
    file_name = Path(file_name)
    if not file_name.is_absolute():
        file_name = Path(__file__).parent / "hparams" / file_name
    if not file_name.exists():
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip("\r\n") for line in token_file if line]


def apply_overfit_test(hparams, dataset):
    """Helper for applying an overfit test conditionally based
    on hyperparameters:

    `overfit_test`: whether or not to apply an overfit test
    `overfit_test_sample_count`: the number of samples to use from the
        original dataset
    `overfit_test_epoch_data_count`: the number of samples per epoch

    The function will accept datasets, (train, valid, test) tuples
    or dictionaries of the form:
    {"train": dataset1, "valid": dataset2, "test": dataset3}

    If a tuple or dictionary is used, the training dataset will be of length
    overfit_test_epoch_data_count wheres the evaluation dataset will be of
    length overfit_test_sample_count.

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    dataset: DynamicItemDataset|tuple|dict
        One of the following
        a dataset
        a dictionary ({"train": dataset1, "valid": dataset2, "test": dataset3})
        a (train, valid, test)  tuple of datasets

    Returns
    -------
    result: DynamicItemDataset|tuple|dict
        a dataset or collection of datasets suitable for
        an overfitting test - in the same format as the
        dataset argument (single dataset, dictionary and tuple)
    """
    if hparams["overfit_test"]:
        if isinstance(dataset, tuple):
            dataset_train, dataset_valid, _ = dataset
            dataset_train = apply_overfit_test(hparams, dataset_train)
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            dataset_eval.set_output_keys(
                list(dataset_valid.pipeline.output_mapping.keys())
            )
            result = dataset_train, dataset_eval, dataset_eval
        elif isinstance(dataset, dict):
            dataset_train = apply_overfit_test(hparams, dataset["train"])
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            dataset_eval.set_output_keys(
                list(dataset["valid"].pipeline.output_mapping.keys())
            )

            result = {
                "train": dataset_train,
                "valid": dataset_eval,
                "test": dataset_eval,
                "sample": dataset_eval,
            }
        else:
            result = dataset.overfit_test(
                hparams["overfit_test_sample_count"],
                hparams["overfit_test_epoch_data_count"],
            )
    else:
        result = dataset
    return result


def undo_padding_tensor(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.

    Arguments
    ---------
    batch : torch.Tensor
        Batch of sentences gathered in a batch.
    lengths : torch.Tensor
        Relative length of each sentence in the batch.

    Returns
    -------
    as_list : list
        A python list of the corresponding input tensor.

    Example
    -------
    >>> batch=torch.rand([4,100])
    >>> lengths=torch.tensor([0.5,0.6,0.7,1.0])
    >>> snt_list=undo_padding(batch, lengths)
    >>> len(snt_list)
    4
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true)
    return as_list


RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        yaml = fin.read()

    # Load evaluation hyperparameters
    eval_hparams_file = Path(hparams_file).parent / "eval.yaml"
    if eval_hparams_file.exists():
        logger.info(
            "Using evaluation hyperparameters from %s", eval_hparams_file
        )
        with open(eval_hparams_file) as eval_hparams:
            hparams_yaml = eval_hparams.read()
            yaml = "\n".join([yaml, hparams_yaml])
    else:
        logger.info(
            "%s not found - not using evaluation hyperparameters",
            eval_hparams_file,
        )
    hparams = load_hyperpyyaml(yaml, overrides, overrides_must_match=True)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    from libritts_prepare import prepare_libritts

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_libritts,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_json_train": hparams["train_json"],
                "save_json_valid": hparams["valid_json"],
                "save_json_test": (
                    hparams["test_json"]
                    if "test" in hparams["splits"]
                    else None
                ),
                "sample_rate": hparams["sample_rate"],
                "train_split": hparams["train_split"],
                "valid_split": hparams["valid_split"],
                "test_split": (
                    hparams["test_split"]
                    if "test" in hparams["splits"]
                    else None
                ),
                "seed": hparams["seed"],
                "model_name": hparams["model"].__class__.__name__,
            },
        )

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)
    audio_keys = ["audio_tokens"]

    # Trainer initialization
    tts_brain = VALLEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    if hparams["testing"]:
        tts_brain.evaluate(
            test_set=datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
