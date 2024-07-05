"""Evaluates a checkpoint using an MOS estimation tool

Authors
* Artem Ploujnikov 2024
"""

#TODO: There are too many evaluation scripts. Refactor to extract common
# features

import speechbrain as sb
import json
import logging
import math
import sys
import csv
import torch
import string
import re
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from types import SimpleNamespace
from torch.nn import ModuleDict
from tqdm.auto import tqdm
from benchmarks.DASB.utils.data import undo_batch
from benchmarks.DASB.utils.eval import vocoder_to_device, Tracker
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class TokotronEvaluator:
    """An evaluator class for the TTS model
    
    Arguments
    ---------
    hparams: dict
        hyperparameters (as a dictionary)
    device : str | torch.device
        the device
    """
    def __init__(self, hparams, device):
        self.hparams = SimpleNamespace(**hparams)
        self.device = device
        modules = self.hparams.modules
        self.modules = ModuleDict(modules).to(self.device)
        suffix = f"_{self.hparams.eval_suffix}" if self.hparams.eval_suffix else ""
        eval_folder = f"eval_{self.hparams.eval_dataset}{suffix}"
        self.output_folder = Path(self.hparams.output_folder) / eval_folder
        self.samples_folder = self.output_folder / "samples"
        self.samples_folder.mkdir(parents=True, exist_ok=True)
        self.spk_emb_model = self.hparams.spk_emb_model(
            run_opts={"device": device}
        )
        self.modules.model.vocoder = None
        self.vocoder_has_details = hasattr(
            self.modules.vocoder,
            "decode_batch_with_details"
        )
        self.enabled_evaluators = set(self.hparams.evaluations.split(","))
        evaluators = hparams.get("evaluators", {})
        if evaluators:
            self.evaluators = {
                key: evaluator_f(run_opts={"device": device})
                for key, evaluator_f in evaluators.items()
                if key in self.enabled_evaluators
            }
        else:
            self.evaluators = {}

        bulk_evaluators = getattr(self.hparams, "bulk_evaluators", {})
        if bulk_evaluators:
            self.bulk_evaluators = {
                key: evaluator_f()
                for key, evaluator_f in bulk_evaluators.items()
                if key in self.enabled_evaluators
            }
        else:
            self.bulk_evaluators = {}

        if not self.evaluators and not self.bulk_evaluators:
            logger.warn("No evaluators were defined - this run will produce samples only")

        self.attention = []
        self.compression = getattr(self.hparams, "compression", False)
        if self.compression:
            self.compression_model = self.hparams.compression_model(
                run_opts={"device": self.device}
            )
            self.modules.model.compression_model = self.compression_model

    def evaluate(self, dataset):
        """Runs evaluation on a dataset

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            a dataset
        """
        logger.info("Recovering the checkpoint")
        ckpt = self.hparams.checkpointer.recover_if_possible()
        if not ckpt:
            raise ValueError("Unable to recover the checkpoint")
        self.modules.model.eval()
        self.tracker = Tracker(
            file_name=self.get_tracker_file_name()
        )
        if self.hparams.eval_samples is not None:
            dataset = dataset.filtered_sorted(select_n=self.hparams.eval_samples)
        dataset = self.tracker.filter(dataset)
        loader = sb.dataio.dataloader.make_dataloader(dataset, batch_size=self.hparams.batch_size)
        loader_it = iter(loader)
        self.create_reports()
        self.modules.model.show_inference_progress = False
        self.item_ids = self.tracker.get_processed()
        details_keys = list(self.evaluators.keys()) + list(self.bulk_evaluators.keys())
        self.details = {
            evaluator_key: []
            for evaluator_key in details_keys
        }
        self.read_reports()
        self.sample_text = []
        self.sample_file_names = []
        self.ref_file_names = []
        logger.info("Starting evaluation")
        batch_count = math.ceil(len(dataset) / self.hparams.batch_size)
        for batch in tqdm(loader_it, desc="Evaluation", total=batch_count):
            self.evaluate_batch(batch)
        self.evaluate_bulk()
        self.write_summary()
        if self.vocoder_has_details:
            self.write_attn()
        logger.info("Evaluation done")

    def create_reports(self):
        """Creates report files and report writers"""
        self.report_files = {}
        self.report_writers = {}
        for evaluator_key in self.enabled_evaluators:
            columns = self.get_report_columns(evaluator_key)
            file_name = self.output_folder / f"{evaluator_key}.csv"
            resume = file_name.exists() and file_name.stat().st_size > 0
            report_file = open(file_name, "a+")
            self.report_files[evaluator_key] = report_file
            writer = csv.DictWriter(report_file, columns)
            if not resume:
                writer.writeheader()
            self.report_writers[evaluator_key] = writer

    def read_reports(self):
        """Invoked when resuming"""
        for evaluator_key in self.enabled_evaluators:
            file_name = self.output_folder / f"{evaluator_key}.csv"
            if file_name.exists():
                logger.info("%s exists, reading")
                with open(file_name) as report_file:
                    reader = csv.DictReader(report_file)
                    for row in reader:
                        del row["uttid"]
                        row = {key : handle_number(value) for key, value in row.items()}
                        self.details[evaluator_key].append(row)

    def get_tracker_file_name(self):
        """Determines the file name of the tracker file"""
        suffix = f"_{self.hparams.eval_suffix}" if self.hparams.eval_suffix else ""
        file_name = f"tracker_{self.hparams.eval_dataset}{suffix}.txt"
        return self.output_folder / file_name

    def get_report_columns(self, evaluator_key):
        """Returns the columns for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            the identifier of the evaluator

        Returns
        -------
        columns : list[str]
            a list of column headers
        """
        bogus_wavs = torch.randn(2, 10000, device=self.device)
        bogus_length = torch.tensor([1., 1.], device=self.device)
        if evaluator_key in self.evaluators:
            evaluator = self.evaluators[evaluator_key]
            result = evaluator.evaluate(
                wavs=bogus_wavs,
                length=bogus_length,
                text=["BOGUS"] * len(bogus_wavs),
                wavs_ref=bogus_wavs,
                length_ref=bogus_length,
            )
        else:
            bogus_file_name = self.output_folder / "bogus.wav"
            evaluator = self.bulk_evaluators[evaluator_key]
            sb.dataio.dataio.write_audio(
                str(bogus_file_name),
                bogus_wavs[0].cpu(),
                samplerate=self.hparams.model_sample_rate,
            )
            result = evaluator.evaluate_files(
                file_names=[bogus_file_name],
                text=["BOGUS"],
                file_names_ref=[bogus_file_name],
            )

        return ["uttid"] + list(result.details.keys())

    def evaluate_batch(self, batch):
        """Runs evaluation on a single batch of speech

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch to be evaluated"""
        with torch.no_grad():
            batch = batch.to(self.device)
            tokens, tokens_length = batch.tokens
            vocoder_to_device(self.modules.vocoder, self.device)
            if hasattr(self.modules.vocoder, "device"):
                self.modules.vocoder.device = self.device
            mel_spec = mel_spec = self.spk_emb_model.mel_spectogram(
                audio=batch.sig.data
            )
            spk_emb = self.spk_emb_model.encode_mel_spectrogram_batch(
                mel_spec, batch.sig.lengths
            ).squeeze(1)
            infer_out = self.modules.model.infer(
                input_tokens=tokens, input_length=tokens_length,
                emb={
                    "spk": spk_emb
                }
            )
            if self.vocoder_has_details:
                wav, details = self.modules.vocoder.decode_batch_with_details(
                    infer_out.audio,
                )
                length = infer_out.length
            else:
                result = self.modules.vocoder(
                    infer_out.audio,
                    infer_out.length,
                )
                if torch.is_tensor(result):
                    wav, length = result, infer_out.length
                else:
                    wav, length = result
                details = {}
            if wav.dim() > 2:
                wav = wav.squeeze(1)
            if "attn" in details:
                self.attention.append(details["attn"])

            self.save_samples(batch, wav, infer_out.length)
            self.item_ids.extend(batch.uttid)
            for evaluator_key, evaluator in self.evaluators.items():
                result = evaluator.evaluate(
                    wavs=wav,
                    length=length,
                    text=batch.label_norm_eval,
                    wavs_ref=batch.sig.data,
                    length_ref=batch.sig.lengths,
                    sample_rate_ref=self.hparams.sample_rate,
                    sample_rate=self.hparams.model_sample_rate
                )
                details = undo_batch(result.details)
                self.write_result(evaluator_key, batch.uttid, details)
                self.details[evaluator_key].extend(details)
            self.tracker.mark_processed(batch.uttid)

    def evaluate_bulk(self):
        """Performs bulk evaluation"""
        for evaluator_key, evaluator in self.bulk_evaluators.items():
            result = evaluator.evaluate_files(
                file_names=self.sample_file_names,
                text=self.sample_text,
                file_names_ref=self.ref_file_names,
            )
            self.details[evaluator_key].append(result.details)
            details = undo_batch(result.details)
            self.write_result(evaluator_key, self.item_ids, details)

    def write_result(self, evaluator_key, uttid, details):
        """Outputs the result details to the report for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        batch : list
            The list of IDs
        details : list
            a list of evaluation details, one dictionary per item
        """
        writer = self.report_writers[evaluator_key]
        for uttid, details_item in zip(uttid, details):
            report_details = {
                "uttid": uttid,
                **details_item,
            }
            writer.writerow(
                ascii_only(flatten(report_details))
            )
        self.report_files[evaluator_key].flush()

    def save_samples(self, batch, wav, length):
        """Saves the samples generated by the TTS system

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch being evaluated
        wav : torch.Tensor
            the waveform
        length: torch.Tensor
            relative lengths
        """
        wav_length_abs = (length * wav.size(1)).int()
        for item_id, infer_wav, wav_length in zip(
            batch.uttid, wav, wav_length_abs
        ):
            file_name = str(
                self.samples_folder / f"{item_id}_pred.wav"
            )
            infer_wav_cut = infer_wav[:wav_length.item()].cpu()
            sb.dataio.dataio.write_audio(
                file_name, infer_wav_cut, samplerate=self.hparams.model_sample_rate
            )
            self.sample_file_names.append(file_name)

    def write_summary(self):
        """Outputs summarized statistics"""
        summary = self.compute_summary()
        file_name = self.output_folder / "summary.json"
        with open(file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)

    def write_attn(self):
        """Outputs attention details"""
        attn_file_name = self.output_folder / "attn.pt"
        torch.save(self.attention, attn_file_name)
        attn_summary_file_name = self.output_folder / "attn_summary.json"
        attn_concat = torch.cat(
            self.attention,
            dim=0
        )
        attn_average = attn_concat.squeeze(-1).mean(0)
        attn_summary_data = {
            "layers": {
                idx + 1 : value.item()
                for idx, value in enumerate(attn_average)
            }
        }
        with open(attn_summary_file_name, "w") as attn_summary_file:
            json.dump(attn_summary_data, attn_summary_file, indent=4)

    def compute_summary(self):
        """Computes the summarized statistics"""
        return {
            f"{evaluator_key}_{stat_key}": value
            for evaluator_key in self.enabled_evaluators
            if evaluator_key in self.details
            for metric_key in self.hparams.eval_summary[evaluator_key]["descriptive"]
            for stat_key, value in descriptive_statistics(
                items=self.details[evaluator_key],
                key=metric_key,
            ).items()
        }


def flatten(value):
    """Converts tensors to scalars and lists of strings to strings

    Arguments
    ---------
    value : dict
        the dictionary to flatten

    Returns
    -------
    result : dict
        a flattened dictionary
    """
    return {
        key: item_value.item() if torch.is_tensor(item_value) else item_value
        for key, item_value in value.items()
    }


RE_PUNCTUATION = re.compile(
    "|".join(
        re.escape(char) for char in string.punctuation
    )
)

RE_NON_ASCII = re.compile(r'[^\x00-\x7F]+')


def ascii_only(values):
    return {
        key: RE_NON_ASCII.sub('', value) if isinstance(value, str)
        else value
        for key, value in values.items()
    }


@sb.utils.data_pipeline.takes("label_norm")
@sb.utils.data_pipeline.provides("label_norm_eval")
def label_norm_pipeline(label):
    """Normalizes labels for ASR comparison, converting to uppercase and removing
    punctuation

    Arguments
    ---------
    label : str
        The unnormalized label

    Returns
    -------
    result : str
        The normalized label
    """
    label = label.upper()
    label = RE_PUNCTUATION.sub("", label)
    return label


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_ref_pipeline(wav):
    """The audio loading pipeline for references

    Arguments
    ---------
    wav : str
        The file path

    Returns
    -------
    sig : torch.Tensor
        The waveform
    """
    sig = sb.dataio.dataio.read_audio(wav)
    return sig


def descriptive_statistics(items, key):
    """Computes descriptive statistics for the summary
    
    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        """
    values = torch.tensor([item[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{key}_{stat_key}": value.item()
        for stat_key, value in stats.items()
    }


def select_subset(dataset, hparams):
    """Selects a subset of the dataset provided, if specified.
    The selection is controlled by a hyperparameter named
    eval_subset, which is expected to list the IDs of the
    data items on which evaluation will take place, one per line

    Arguments
    ---------
    dataset : speechbrain.dataio.dataset.DynamicItemDataset
        A dataset
    hparams : dict
        A hyperparameters file

    Returns
    -------
    subset : dataset
        The dataset, filtered down if applicable
    """
    eval_subset_path = hparams.get("eval_subset")
    if eval_subset_path is not None:
        eval_subset_path = Path(eval_subset_path)
        if not eval_subset_path.exists():
            raise ValueError(f"eval_subset {eval_subset_path} does not exist")
        with open(eval_subset_path) as eval_subset_file:
            eval_subset_ids = [line.strip() for line in eval_subset_file]
        subset = FilteredSortedDynamicItemDataset(dataset, eval_subset_ids)
    else:
        subset = dataset
    return subset


RE_INTEGER = re.compile(r"^-?\d+$")
RE_FLOAT = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def handle_number(value):
    """Converts a value to a number, if applicable"""
    if RE_INTEGER.match(value):
        value = int(value)
    elif RE_FLOAT.match(value):
        value = float(value)
    return value


if __name__ == "__main__":
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Reuse the preparation function from the training script
    from train import dataio_prepare

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides, overrides_must_match=False)

    # Load evaluation hyperparameters
    eval_hparams_file = hparams.get("eval_hparams")
    if eval_hparams_file is None:
        # If not defined, look for eval.yaml in the same folder
        # as the original hyperparameters file
        eval_hparams_file = Path(hparams_file).parent / "eval.yaml"
    if eval_hparams_file.exists():
        logger.info(
            "Using evaluation hyperparameters from %s",
            eval_hparams_file
        )
        eval_hparams = load_hyperpyyaml(
            eval_hparams_file, overrides, overrides_must_match=False
        )
        hparams.update(eval_hparams)
    else:
        logger.info(
            "%s not found - not using evaluation hyperparameters",
            eval_hparams_file
        )

    # Data Preparation
    from libritts_prepare import prepare_libritts

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        with hparams["freezer"]:
            extract_features = []
            if hparams["input"] == "phonemes":
                extract_features.append("phn")
            run_on_main(
                prepare_libritts,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_folder": hparams["prepare_save_folder"],
                    "save_json_train": hparams["train_json"],
                    "save_json_valid": hparams["valid_json"],
                    "save_json_test": (
                        hparams["test_json"] if "test" in hparams["splits"]
                        else None
                    ),
                    "sample_rate": hparams["sample_rate"],
                    "train_split": hparams["train_split"],
                    "valid_split": hparams["valid_split"],
                    "test_split": (
                        hparams["test_split"] if "test" in hparams["splits"]
                        else None
                    ),
                    "extract_features": extract_features,
                    "seed": hparams["seed"],
                    "extract_features_opts": hparams["extract_features_opts"],
                    "model_name": hparams["model"].__class__.__name__,
                    "device": run_opts.get("device", "cpu"),
                },
            )

    # Reading command line arguments
    hparams["guides_enabled"] = False
    datasets, _ = dataio_prepare(hparams)

    # Select the dataset to use in evaluation
    eval_dataset_key = hparams.get("eval_dataset", "valid")

    eval_dataset = datasets[eval_dataset_key]
    eval_dataset = select_subset(eval_dataset, hparams)
    eval_dataset.add_dynamic_item(label_norm_pipeline)
    eval_dataset.add_dynamic_item(audio_ref_pipeline)
    eval_dataset.set_output_keys(
        ["uttid", "label_norm_eval", "tokens", "sig"]
    )

    # Create the evaluator
    eval = TokotronEvaluator(hparams, device=run_opts["device"])

    # Start evaluation
    logger.info("Evaluating on %s", eval_dataset_key)
    eval.evaluate(eval_dataset)
