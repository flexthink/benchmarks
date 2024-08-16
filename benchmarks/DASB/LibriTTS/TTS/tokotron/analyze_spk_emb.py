import csv
import logging
import speechbrain as sb
import torch
import torchaudio
import sys
from tqdm.auto import tqdm
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.distributed import run_on_main
from benchmarks.DASB.model.Tokotron import RepresentationMode
from types import SimpleNamespace


logger = logging.getLogger(__name__)


class SpkEmbAnalysis:
    """Performs speaker embedding analysis

    Arguments
    ---------
    hparams : dict
        Raw hyperparameters
    """
    def __init__(self, hparams, run_opts=None):
        self.hparams = SimpleNamespace(**hparams)
        if run_opts is None:
            run_opts = {}
        self.device = run_opts.get("device", "cpu")

    def analyze(self, dataset):
        logger.info("Beginning analysis")
        loader = sb.dataio.dataloader.make_dataloader(dataset, batch_size=self.hparams.batch_size)
        loader_it = iter(loader)
        self.on_analyze_start()
        for batch in tqdm(loader_it, total=len(dataset) // self.hparams.batch_size, desc="Computing embeddings"):
            self.analyze_batch(batch)
        self.on_analyze_end()

    def on_analyze_start(self):
        """Invoked at teh beginning of analysis"""
        self.embs = []
        self.spk_id = []
        self.uttid = []
        self.output_folder = Path(self.hparams.output_folder) / "spk_emb"
        logger.info("Saving analysis data in %s", self.output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.eval_spk_sim = self.hparams.eval_spk_sim(
            run_opts={"device": self.device}
        )
        self.spk_emb_model = self.eval_spk_sim.model

    @torch.no_grad()
    def analyze_batch(self, batch):
        """Processes a single batch

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            a padded batch
        """
        batch = batch.to(self.device)
        wav = torchaudio.functional.resample(
            batch.sig.data,
            orig_freq=self.hparams.sample_rate,
            new_freq=self.hparams.eval_spk_sim_sample_rate
        )
        length_abs = batch.sig.lengths * wav.size(1)
        attention_mask = length_to_mask(length_abs.int()).long()
        result = self.spk_emb_model(
            input_values=wav,
            attention_mask=attention_mask
        )
        self.embs.append(result.embeddings.cpu())
        self.spk_id.extend(batch.spk_id)
        self.uttid.extend(batch.uttid)

    def on_analyze_end(self):
        """Involed at the end of analysis"""
        logger.info("Extraction done, performing final analysis")
        self.reorganize()
        self.save_raw()
        self.compute_stats()
        self.save_stats()

    def compute_stats(self):
        """Computes statistics from speaker embeddings"""
        self.spk_id_map = self.group_speakers()
        self.spk_id_unique = sorted(set(self.spk_id))
        self.embs_grouped = self.group_embs()
        self.centroids = self.compute_centroids()
        self.similarities = self.compute_similarities()
        self.similarities_grouped = self.group_similarities()
        (
            self.similarities_stats,
            self.similarities_stats_grouped
        ) = self.describe_similarities()
        self.cross_similarity = self.compute_cross_similarity()

    def save_stats(self):
        """Saves computed statistics"""
        self.save_similarities()
        self.save_similarities_grouped()
        self.save_similarities_stats()
        self.save_cross_similarity()

    def reorganize(self):
        """Reorganizes representations to remove any artifacts of batching"""
        # Concatenate along the batch dimension
        self.embs = torch.cat(
            self.embs,
            dim=0
        )

    def save_raw(self):
        """Saves a raw representation"""
        file_name = self.output_folder / "raw.pt"
        details = {
            "embs": self.embs,
            "spk_id": self.spk_id,
            "uttid": self.uttid,
        }
        with open(file_name, "wb") as embs_file:
            torch.save(
                details,
                embs_file
            )

    def group_speakers(self):
        """Creates a grouping dictionary of spk_id -> list of indices
        
        Returns
        -------
        spk_id_map : dict
            the mapping"""
        spk_id_map = {}
        for idx, spk_id in enumerate(self.spk_id):
            if spk_id not in spk_id_map:
                spk_id_map[spk_id] = []
            spk_id_map[spk_id].append(idx)
        return spk_id_map

    def group_embs(self):
        """Groups embeddings by speaker
        
        Returns
        -------
        embs_grouped : dict
            A dictionary of embeddings, grouped by speaker
        """
        embs_grouped = {
            spk_id: self.embs[indexes]
            for spk_id, indexes in self.spk_id_map.items()
        }
        return embs_grouped

    def compute_centroids(self):
        """Computes the centroid of the embedding cluster for each speaker"""
        centroids = {
            spk_id: embs.mean(dim=0)
            for spk_id, embs in self.embs_grouped.items()
        }
        return centroids

    def compute_similarities(self):
        """Computes the cosine similarity between each individual embedding and its centroid"""
        centroids = torch.stack(
            [
                self.centroids[spk_id]
                for spk_id in self.spk_id
            ]
        )
        similarities = torch.nn.functional.cosine_similarity(
            self.embs,
            centroids,
            dim=-1
        )
        return similarities

    def group_similarities(self):
        """Groups similarity scores by speaker ID"""
        similarities_grouped = {
            spk_id: self.similarities[indexes]
            for spk_id, indexes in self.spk_id_map.items()
        }
        return similarities_grouped

    def save_similarities(self):
        """Saves raw similarities"""
        file_name = self.output_folder / "sim.csv"
        with open(file_name, "w") as sim_csv_file:
            sim_csv_writer = csv.DictWriter(
                sim_csv_file,
                fieldnames=["uttid", "spk_id", "sim"]
            )
            sim_csv_writer.writeheader()
            for uttid, spk_id, sim in zip(self.uttid, self.spk_id, self.similarities):
                sim_csv_writer.writerow(
                    {
                        "uttid": uttid,
                        "spk_id": spk_id,
                        "sim": sim.item(),
                    }
                )

    def save_similarities_grouped(self):
        """Saves the similarities by speaker"""
        file_name = self.output_folder / "sim_stats_grouped.csv"
        with open(file_name, "w") as sim_csv_file:
            sim_csv_file = csv.DictWriter(
                sim_csv_file,
                fieldnames=["spk_id", "sim"]
            )
            sim_csv_file.writeheader()
            for spk_id, sim in self.similarities_grouped.items():
                sim_csv_file.writerow(
                    {
                        "spk_id": spk_id,
                        "sim": sim.mean(dim=0).item()
                    }
                )

    def describe_similarities(self):
        """Describes cosine similarities"""
        stats = descriptive_statistics(
            self.similarities
        )
        stats_grouped = {
            spk_id: descriptive_statistics(similarities)
            for spk_id, similarities
            in self.similarities_grouped.items()
        }
        return stats, stats_grouped

    def save_similarities_stats(self):
        """Save statistics about similarity"""
        columns = ["spk_id"] + list(self.similarities_stats.keys())
        file_name = self.output_folder / "sim_stats.csv"
        with open(file_name, "w") as stats_file:
            stats_writer = csv.DictWriter(
                stats_file,
                fieldnames=columns
            )
            stats_writer.writeheader()
            stats_writer.writerow(
                {
                    "spk_id": "all",
                    **self.similarities_stats
                }
            )
            for spk_id, spk_stats in self.similarities_stats_grouped.items():
                stats_writer.writerow(
                    {
                        "spk_id": spk_id,
                        **spk_stats
                    }
                )

    def compute_cross_similarity(self):
        """Computes the cross-similarity of centroids

        Returns
        -------
        cross_sim : torch.Tensor
            A symmetric tensor in which [i, j] is the cosine similarity
            between the centroids of the ith and the jth speaker"""
        spk_ids = self.spk_id_unique
        emb = torch.cat(
            [
                self.centroids[spk_id][None, :].repeat(len(spk_ids), 1)
                for spk_id in spk_ids
            ],
            dim=0
        )
        ref = torch.stack(
            [
                self.centroids[spk_id]
                for spk_id in spk_ids
            ]
        ).repeat(len(spk_ids), 1)
        sim = torch.nn.functional.cosine_similarity(
            emb,
            ref,
            dim=-1
        )
        cross_sim = sim.reshape(len(spk_ids), len(spk_ids))
        return cross_sim

    def save_cross_similarity(self):
        """Saves a cross-similarity table"""
        file_name = self.output_folder / "cross_similairty.csv"
        with open(file_name, "w") as cross_file:
            cross_writer = csv.writer(cross_file)
            cross_writer.writerow(["spk_id"] + self.spk_id_unique)
            for spk_id, row in zip(self.spk_id_unique, self.cross_similarity):
                cross_writer.writerow(
                    [spk_id] + row.tolist()
                )
       

def descriptive_statistics(values):
    """Computes simple descriptive statistics of a tensor

    Arguments
    ---------
    values : torch.Tensor
        The values to aggregate

    Returns
    -------
    stats : dict
        The statistics
    """
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats_t = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    stats = {key: value.item() for key, value in stats_t.items()}
    return stats    

@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(wav):
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
    representation_mode = RepresentationMode(
        hparams.get("representation_mode", RepresentationMode.DISCRETE)
    )
    audio_features = "audio_tokens" if representation_mode == RepresentationMode.DISCRETE else "audio_ssl"
    if not hparams["skip_prep"]:
        with hparams["freezer"]:
            extract_features = [audio_features, "spk_emb"]
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

    # We can now directly create the datasets for training, valid, and test
    from train import dataio_prepare
    (
        datasets,
        silence_padding,
        resample_fn
    ) = dataio_prepare(hparams)

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

    eval_dataset_key = hparams.get("eval_dataset", "valid")
    eval_dataset = datasets[eval_dataset_key]
    eval_dataset.add_dynamic_item(audio_pipeline)
    eval_dataset.set_output_keys(["uttid", "spk_id", "sig"])

    logger.info("Performing speaker embedding analysis on the %s dataset")
    analysis = SpkEmbAnalysis(
        hparams=hparams,
        run_opts=run_opts,
    )
    analysis.analyze(eval_dataset)
