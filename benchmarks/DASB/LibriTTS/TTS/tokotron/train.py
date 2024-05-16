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
import math
import torch
import torchaudio
import sys
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.utils.distributed import run_on_main
from benchmarks.DASB.utils.preparation import add_prepared_features
from benchmarks.DASB.utils.audio_tokens import (
    get_silence_token,
    use_silence_padding,
    feature_pad_to,
)
from benchmarks.DASB.utils.curriculum import CurriculumSpeechDataset, SampleMode

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_COUNT = 1


# Brain class for speech recognition training
class TokotronBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

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
        tokens, tokens_length = batch.tokens
        input_offset, audio_offset = self.compute_offsets(batch)
        audio_tokens, audio_tokens_length = batch.audio_tokens_bos
        if self.compression:
            audio_tokens = self.compression_model.compress(audio_tokens)
        predictions = self.modules.model(
            input_tokens=tokens,
            input_length=tokens_length,
            audio_tokens=audio_tokens,
            audio_length=audio_tokens_length,
            input_offset=input_offset,
            audio_offset=audio_offset,
            emb={
                "spk": batch.spk_emb.data.squeeze(1)
            }
        )

        return predictions

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
        audio_tokens, audio_tokens_length = batch.audio_tokens_pad
        if self.compression:
            audio_tokens = self.compression_model.compress(audio_tokens)
        loss_details = self.hparams.compute_cost(
            predictions=predictions,
            audio_tokens=audio_tokens,
            audio_length=audio_tokens_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
        )
        self.loss_metric.append(
            batch.uttid,
            predictions=predictions,
            audio_tokens=audio_tokens,
            audio_length=audio_tokens_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
            reduction="batch",
        )
        return loss_details.loss

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
        if hasattr(self.modules.vocoder, "model"):
            self.modules.vocoder.model.device = self.device
        self.create_perfect_samples()
        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost, batch_eval=True,
        )
        if (
            self.hparams.audio_emb_pretrained
            and epoch == 1
            and stage == sb.Stage.TRAIN
        ):
            # TODO: Clean this up
            if hasattr(self.hparams.token_model, "vocabulary"):
                vocabulary = self.hparams.token_model.vocabulary
            elif hasattr(self.hparams.token_model, "vocabularies"):
                vocabulary = torch.stack(
                    [
                        torch.from_numpy(voc)
                        for voc in self.hparams.token_model.vocabularies
                    ]
                )
            self.modules.model.init_audio_emb(vocabulary)
        
        # Curriculum Learning
        if self.hparams.curriculum_enabled and not self.hparams.overfit_test:
            self.set_curriculum(stage, epoch)

        # Load the compression model only if compression is enabled
        self.compression = getattr(self.hparams, "compression", False)
        if self.compression:
            self.compression_model = self.hparams.compression_model(
                run_opts={"device": self.device}
            )
            self.modules.model.compression_model = self.compression_model

    def set_curriculum(self, stage, epoch):
        """Sets up curriculum learning

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        stage_key = stage.name.lower()
        step_id, step = self.hparams.curriculum[stage_key].apply(epoch)
        sample_mode = SampleMode(
            step.get("sample_mode", SampleMode.SEGMENT)
        )
        if sample_mode == SampleMode.FULL:
            logger.info(
                "%s: Curriculum step %d, using sampling full sentences, %s samples",
                stage.name,
                step_id,
                step.get("num_samples"),
            )
        else:
            logger.info(
                "%s: Curriculum step %d, using sampling with %s-%s words, %s samples",
                stage.name,
                step_id,
                step.get("min_words", 0),
                step.get("max_words", "unlimited"),
                step.get("num_samples", "unlimited"),
            )

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

            self.create_samples()

    def compute_offsets(self, batch):
        """Computes offsets for curriculum learning

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.

        Returns
        -------
        input_offset : torch.Tensor
            Positional offset for tokens
        audio_offset : torch.Tensor
            Positional offsets for audio
        """
        input_offset = self.compute_rel_offset(
            item=batch.tokens,
            rel_length=batch.char_rel_length,
            rel_offset=batch.char_rel_offset,
        )
        audio_offset = self.compute_rel_offset(
            item=batch.audio_tokens,
            rel_length=batch.sig_rel_length,
            rel_offset=batch.sig_rel_offset,
        )
        return input_offset, audio_offset

    def compute_rel_offset(self, item, rel_length, rel_offset, bos_len=1):
        len_abs = item.lengths * item.data.size(1)
        len_abs_bos = len_abs + bos_len
        return ((rel_offset / rel_length) * len_abs + bos_len) / len_abs_bos

    def fit_batch(self, batch):
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss

    def create_samples(self):
        """Writes audio samples at the end of an epoch"""
        epoch = self.hparams.epoch_counter.current
        if epoch % self.hparams.samples_interval != 0:
            return
        if self.debug:
            self.modules.model.decoder.infer_max_decoder_steps = (
                self.hparams.debug_infer_max_audio_length
            )
        sample_loader = sb.dataio.dataloader.make_dataloader(
            self.sample_data, **self.hparams.sample_dataloader_opts,
        )
        for batch in sample_loader:
            batch = batch.to(self.device)
            tokens, tokens_length = batch.tokens
            infer_out = self.modules.model.infer(
                input_tokens=tokens,
                input_length=tokens_length,
                emb={
                    "spk": batch.spk_emb.data.squeeze(1)
                }
            )
            self.hparams.progress_report.write(
                ids=batch.uttid,
                audio=infer_out.wav,
                length_pred=infer_out.wav_length,
                length=batch.audio_tokens_pad.lengths,
                alignments=infer_out.alignments,
                p_eos=infer_out.p_eos,
            )

    def create_perfect_samples(self):
        """Creates the best samples that can be created using
        the vocoder provided, for comparison purposes"""
        if not self.hparams.progress_logger["perfect_samples_created"]:
            sample_loader = sb.dataio.dataloader.make_dataloader(
                self.sample_data, **self.hparams.sample_dataloader_opts
            )
            for batch in sample_loader:
                batch = batch.to(self.device)
                sample_tokens, length = batch.audio_tokens_pad
                vocoder_out = self.modules.vocoder(
                    sample_tokens,
                    length,
                    spk=batch.spk_emb.data.squeeze(1)
                )
                if isinstance(vocoder_out, tuple):
                    samples, samples_length = vocoder_out
                else:
                    samples = vocoder_out
                    samples_length = length
                if samples.dim() == 3:
                    samples = samples.squeeze(1)
                max_len = samples.size(1)
                samples_length_abs = (samples_length * max_len).int()
                with self.hparams.progress_logger:
                    for item_id, item_wav, item_length in zip(
                        batch.uttid, samples, samples_length_abs
                    ):
                        item_cut = item_wav[: item_length.item()]
                        self.hparams.progress_logger.save(
                            name=f"{item_id}.wav",
                            content=item_cut.detach().cpu(),
                            mode="audio",
                            folder="_perfect",
                            samplerate=self.hparams.sample_rate,
                        )
                    self.hparams.progress_logger[
                        "perfect_samples_created"
                    ] = True
                    self.hparams.progress_logger.clear()


INPUT_FEATURE_MAP = {"text": "char", "phonemes": "phn"}
OUTPUT_KEYS = [
    "uttid",
    "tokens",
    "audio_tokens",
    "audio_tokens_pad",
    "audio_tokens_bos",
    "spk_emb",
    "char",
    "char_rel_length",
    "char_rel_offset",
    "sig_rel_offset",
    "sig_rel_length",
]


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

    input_feature_key = f"_{input_feature}" if hparams["curriculum_enabled"] else input_feature
    @sb.utils.data_pipeline.takes(input_feature_key)
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label_encoder.encode_sequence_torch(label)

    # TODO: Update curriculum to avoid reading audio
    def resample_audio(sig):
        return torchaudio.functional.resample(
            sig,
            orig_freq=hparams["sample_rate"],
            new_freq=hparams["model_sample_rate"],
        )

    use_silence_padding = hparams.get("use_silence_padding", True)
    if use_silence_padding:
        silence_token, _ = get_silence_token(
            hparams["token_model"],
            extract_emb=False,
            model_kwargs=hparams.get("token_model_kwargs"),
        )
    else:
        silence_token = (
            torch.ones(hparams["audio_tokens_per_step"], dtype=torch.int64)
            * hparams["eos_index"]
        )
    silence_token = silence_token.cpu()
    silence_padding_len = int(math.ceil(hparams["silence_padding"]))
    bos_width = hparams.get("bos_width", 1)
    audio_bos = (
        torch.ones(bos_width, hparams["audio_tokens_per_step"])
        * hparams["bos_index"]
    )

    audio_tokens_key = "_audio_tokens" if hparams["curriculum_enabled"] else "audio_tokens"
    @sb.utils.data_pipeline.takes(audio_tokens_key)
    @sb.utils.data_pipeline.provides("audio_tokens_pad", "audio_tokens_bos")
    def audio_pipeline(audio_tokens):
        audio_tokens = torch.from_numpy(audio_tokens)
        audio_tokens_pad = feature_pad_to(
            audio_tokens, len(audio_tokens) + silence_padding_len, silence_token
        )
        yield audio_tokens_pad
        audio_tokens_bos = torch.cat([audio_bos, audio_tokens_pad], dim=0)
        yield audio_tokens_bos

    dynamic_items = [tokens_pipeline, audio_pipeline]

    init_sequence_encoder(hparams)
    raw_datasets = {}
    for dataset in data_info:
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            output_keys=OUTPUT_KEYS,
        )
        add_prepared_features(
            dataset=dynamic_dataset,
            save_path=Path(hparams["prepare_save_folder"]) / "features",
            id_key="uttid",
            features=["audio_tokens", "spk_emb"],
        )
        raw_datasets[dataset] = dynamic_dataset

        # Use the curriculum sampler to reduce the dataset's complexity
        if hparams.get("curriculum_enabled"):
            curriculum_generator = torch.Generator()
            curriculum_generator.manual_seed(hparams["seed"])
            dynamic_dataset = CurriculumSpeechDataset(
                from_dataset=dynamic_dataset,
                generator=curriculum_generator,
                audio_keys=["audio_tokens"],
                passthrough_keys=["spk_emb"],
                sample_rate=hparams["model_sample_rate"],
                process_audio=resample_audio,
            )
            dynamic_dataset.set_output_keys(OUTPUT_KEYS)
            curriculum = hparams["curriculum"][dataset]
            curriculum.bind(dynamic_dataset)
            curriculum.apply(1)            
        else:
            logger.info(
                "Curriculum sampling is disabled, using the complete dataset"
            )

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
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    if hparams["curriculum_enabled"]:        
        sample_dataset = CurriculumSpeechDataset(
            from_dataset=raw_datasets["valid"],
            generator=curriculum_generator,
            audio_keys=["audio_tokens"],
            passthrough_keys=["spk_emb"],
            sample_rate=hparams["model_sample_rate"],
            process_audio=resample_audio,
        )
        curriculum = hparams["curriculum_sample"]
        curriculum.bind(sample_dataset)
        curriculum.apply(1)
        sample_dataset.set_output_keys(OUTPUT_KEYS)
        datasets["sample"] = sample_dataset
    else:
        datasets["sample"] = select_sample(hparams, datasets)

    for dataset in datasets.values():
        for dynamic_item in dynamic_items:
            dataset.add_dynamic_item(dynamic_item)

    return datasets, silence_token


def select_sample(hparams, datasets):
    """Selects a sample of files for sample generation, freezing the sample if
    requested to persist across multiple experiments

    Arguments
    ---------
    hparams : dict
        experiment hyperparameters
    datasets : dict
        a dictionary of datasets

    Returns
    -------
    dataset : speechbrain.dataio.dataset.FilteredSortedDynamicItemDataset
        the sample dataset
    """
    sample_path = hparams.get("sample_path")
    dataset = None
    if sample_path is not None:
        sample_path = Path(sample_path)
        if sample_path.exists():
            with open(sample_path, "r") as sample_file:
                data_ids = [line.strip() for line in sample_file]
                dataset = FilteredSortedDynamicItemDataset(
                    datasets["valid"], data_ids
                )

    if dataset is None:
        dataset = (
            datasets["valid"]
            .batch_shuffle(1)
            .filtered_sorted(select_n=hparams["num_audio_samples"])
        )
        if sample_path is not None:
            with open(sample_path, "w") as sample_file:
                for data_id in dataset.data_ids:
                    print(data_id, file=sample_file)
    return dataset


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
    if not Path(file_name).exists():
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
            dataset_train, _, _ = dataset
            dataset_train = apply_overfit_test(hparams, dataset_train)
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = dataset_train, dataset_eval, dataset_eval
        elif isinstance(dataset, dict):
            dataset_train = apply_overfit_test(hparams, dataset["train"])
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
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


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from libritts_prepare import prepare_libritts

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        extract_features = ["audio_tokens", "spk_emb"] if not hparams["skip_extract_features"] else None
        with hparams["freezer"]:
            run_on_main(
                prepare_libritts,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "alignments_folder": hparams["data_folder_alignments"],
                    "save_folder": hparams["prepare_save_folder"],
                    "save_json_train": hparams["train_json"],
                    "save_json_valid": hparams["valid_json"],
                    "save_json_test": hparams["test_json"],
                    "sample_rate": hparams["sample_rate"],
                    "train_split": hparams["train_split"],
                    "valid_split": hparams["valid_split"],
                    "test_split": hparams["test_split"],
                    "extract_features": extract_features,
                    "seed": hparams["seed"],
                    "extract_features_opts": hparams["extract_features_opts"],
                    "model_name": hparams["model"].__class__.__name__,
                    "device": run_opts.get("device", "cpu"),
                },
            )

    # We can now directly create the datasets for training, valid, and test
    datasets, silence_token = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)
    token_keys = ["audio_tokens_pad", "audio_tokens_bos"]
    
    # Trainer initialization
    tts_brain = TokotronBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    tts_brain.sample_data = datasets["sample"]

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=use_silence_padding(
            hparams["train_dataloader_opts"], silence_token, token_keys
        ),
        valid_loader_kwargs=use_silence_padding(
            hparams["valid_dataloader_opts"], silence_token, token_keys
        ),
    )

    # Load best checkpoint for evaluation
    test_stats = tts_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        test_loader_kwargs=use_silence_padding(
            hparams["test_dataloader_opts"], silence_token, token_keys
        ),
    )

    # Save final checkpoint (fixed name)
    tts_brain.checkpointer.save_checkpoint(name="latest")
