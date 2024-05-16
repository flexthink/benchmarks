"""
Utilities for curriculum learning

Authors
* Artem Ploujnikov 2022
"""
import os
import torch
import speechbrain as sb
import itertools
from speechbrain.utils import checkpoints
from enum import Enum

from speechbrain.utils.data_pipeline import DataPipeline

from speechbrain.dataio.dataset import (
    DynamicItemDataset,
    FilteredSortedDynamicItemDataset,
)


SAMPLE_OUTPUTS = [
    "wrd_count",
    "sig",
    "sig_rel_offset",
    "sig_rel_length",
    "wrd_start",
    "wrd_end",
    "phn_start",
    "phn_end",
    "wrd",
    "char",
    "char_rel_offset",
    "char_rel_length",
    "phn",
    "phn_rel_offset",
    "phn_rel_length",
    "char_full",
    "phn_full"
]


class SampleMode(Enum):
    SEGMENT = "segment"
    FULL = "full"


class Sorting(Enum):
    RANDOM = "random"
    ASCENDING = "ascending"
    DESCENDING = "descending"


class CurriculumSpeechDataset(DynamicItemDataset):
    """A derivative dynamic dataset that allows to perform
    curriculum learning over a speech dataset with phoneme
    alignments similar to LibriSpeech-Alignments. The dataset
    selects sub-samples within the specified length range in words

    Arguments
    ---------
    from_dataset: DynamicItemDataset
        a base dataset compatible with alignments-enhanced LibriSpeech
    min_words: int
        the minimum number of words to sample from each dataset item
    max_words: int
        the maximum number of words to sample from each dataset item
    num_samples: int
        the number of samples per epoch
    sample_rate: int
        the audio sampling rate, in Hertz
    sample_mode: SampleMode|str
        the smapling mode
        SampleMode.SEGMENT will select segments of varying lengths
        SampleMode.FULL will sample complete data samples, taking only
            num_samples into consideration
    generator: torch.Generator
        a random number generator (optional). A custom generator may
        be supplied for reproducibility or fofr unit tests.
    audio_keys : list, optional
        additional representations of "audio", to be "cut" the same way
        as raw audio, assumed to be scaled linearly, such as audio tokens
        or SSL representations
    passthrough_keys : list, optional
        data from the original dataset that does not depend on length
        and can be passed through (e.g. speaker labels, language labels,
        any other categorical labels, speaker embeddings, etc)
    """

    def __init__(
        self,
        from_dataset,
        min_words=None,
        max_words=None,
        num_samples=None,
        sample_rate=16000,
        sample_mode=SampleMode.SEGMENT,
        sorting=Sorting.RANDOM,
        process_audio=None,
        generator=None,
        audio_keys=None,
        passthrough_keys=None,
    ):

        self.data = from_dataset.data
        self.data_ids = list(self.data.keys())
        static_keys = list(self.data[self.data_ids[0]].keys())
        static_keys.append("id")
        self.audio_keys = audio_keys or []
        self.passthrough_keys = passthrough_keys or []
        self.pipeline = DataPipeline(static_keys)
        self.pipeline.add_dynamic_items(from_dataset.pipeline.dynamic_items)

        self.base_dataset = from_dataset
        self.data_ids = self.base_dataset.data_ids
        self._refresh_index_map()
        self.min_words = min_words
        self.max_words = max_words
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.sample_mode = sample_mode
        self.sorting = Sorting(sorting)
        self._pipeline_is_setup = False
        if (
            min_words is not None
            or max_words is not None
            or num_samples is not None
        ):
            self.sample_segments(generator=generator)
        self.process_audio = process_audio
        self.setup_pipeline()
        pipeline_outputs = SAMPLE_OUTPUTS + self.audio_keys + self.passthrough_keys
        self.pipeline = PipelineWrapper(self.pipeline, pipeline_outputs)

    def _refresh_index_map(self):
        """Refreshes the data ID to index map"""
        self.data_id_indices = {
            data_id: idx for idx, data_id in enumerate(self.data_ids)
        }

    def sample_segments(self, dataset=None, generator=None):
        """Samples parts of the audio file at specific word boundaries

        Arguments
        ---------
        datset: DynamicItemDataset
            the dataset from which to sample
        generator: torch.Generator
            a random number generator (optional)
        """
        # Exclude samples than have fewer
        # words than the minimum
        if not generator:
            generator = torch.default_generator
        if dataset is None:
            dataset = self.base_dataset
        if self.sample_mode == SampleMode.FULL:
            self.sample_full(dataset, generator)
            return
        dataset = sample(dataset, self.num_samples, generator)
        min_words = self.min_words
        if min_words is None:
            min_words = 0
        dataset = dataset.filtered_sorted(
            key_min_value={"wrd_count": min_words},
            sort_key=None if self.sorting == Sorting.RANDOM else "duration",
            reverse=self.sorting == Sorting.DESCENDING
        )
        self.data_ids = dataset.data_ids
        self._refresh_index_map()
        wrd_count, wrd_start, wrd_end = self._word_counts(dataset)

        # Randomly sample word counts in the
        # range form num_words to last_words
        sample_word_counts = torch.randint(
            low=self.min_words,
            high=self.max_words + 1,
            size=(len(dataset),),
            generator=generator,
        )
        sample_word_counts = torch.minimum(sample_word_counts, wrd_count)
        self.sample_word_counts = sample_word_counts

        # Sample relative offsets, from 0.0 to 1.0.
        # 0.0 corresponds to the beginning of the
        # utterance, where as 1.0 represents wrd_count - n
        # where n is the sampled word count
        sample_offsets_rel = torch.rand(len(dataset), generator=generator)

        # Determine the maximum possible offsets
        max_offset = wrd_count - self.sample_word_counts
        self.wrd_offset_start = (
            (sample_offsets_rel * max_offset).floor().int().clamp(0)
        )
        self.wrd_offset_end = self.wrd_offset_start + self.sample_word_counts
        self.wrd_offset_end = torch.maximum(
            self.wrd_offset_end, self.wrd_offset_start + 1
        )
        self.wrd_offset_end = torch.minimum(self.wrd_offset_end, wrd_count - 1)
        sample_start = torch.tensor(
            [item[idx] for item, idx in zip(wrd_start, self.wrd_offset_start)]
        )
        sample_end = torch.tensor(
            [item[idx - 1] for item, idx in zip(wrd_end, self.wrd_offset_end)]
        )
        sample_start_idx = time_to_index(sample_start, self.sample_rate)
        sample_end_idx = time_to_index(sample_end, self.sample_rate)
        self.sample_start_idx = sample_start_idx
        self.sample_end_idx = sample_end_idx

    def sample_full(self, dataset, generator=None):
        """Samples full sentences only 

        Arguments
        ---------
        dataset: DynamicItemDataset
            the dataset from which to sample
        generator: torch.Generator
            a random number generator (optional)

        """
        if dataset is None:
            dataset = self.base_dataset
        dataset = sample(dataset, self.num_samples, generator)
        if self.sorting != Sorting.RANDOM:
            dataset = dataset.filtered_sorted(
                sort_key=None if self.sorting == Sorting.RANDOM else "duration",
                reverse=self.sorting == Sorting.DESCENDING
            )
        self.data_ids = dataset.data_ids
        self.sample_end_idx = None
        self.sample_end_idx = None
        self.wrd_offset_start = None
        self.wrd_offset_end = None
        self.sample_word_counts = None
        self._refresh_index_map()

    def _word_counts(self, dataset):
        keys = ["wrd_count", "wrd_start", "wrd_end"]
        with dataset.output_keys_as(keys):
            wrd_count = torch.tensor(self._pluck("wrd_count"))
            wrd_start = self._pluck("wrd_start")
            wrd_end = self._pluck("wrd_end")
        return wrd_count, wrd_start, wrd_end

    def _pluck(self, key):
        """Retrieves a list of values of the specified key from
        all data items in the dataset

        Arguments
        ---------
        key: str
            the key

        Returns
        -------
        result: list
            the resulting list"""
        return [self.data[data_id][key] for data_id in self.data_ids]

    def setup_pipeline(self):
        """Sets up the dynamic pipeline to sample from the dataset
        using the previously generated samples"""
        import torchaudio

        @sb.utils.data_pipeline.takes(
            "id",
            "wav",
            "wrd_count",
            "wrd_start",
            "wrd_end",
            "phn_start",
            "phn_end",
            "wrd",
            "phn",
            *self.audio_keys,
            *self.passthrough_keys
        )
        @sb.utils.data_pipeline.provides(
            "_wrd_count",
            "_sig",
            "_sig_rel_offset",
            "_sig_rel_length",
            "_wrd_start",
            "_wrd_end",
            "_phn_start",
            "_phn_end",
            "_wrd",
            "_char",
            "_char_rel_offset",
            "_char_rel_length",
            "_phn",
            "_phn_rel_offset",
            "_phn_rel_length",
            "_char_full",
            "_phn_full",
            *[f"_{key}" for key in self.audio_keys],
            *[f"_{key}" for key in self.passthrough_keys],
        )
        def cut_sample(
            data_id, wav, wrd_count, wrd_start, wrd_end, phn_start, phn_end, wrd, phn, *args
        ):
            idx = self.data_id_indices[data_id]
            # wrd_count
            sig = sb.dataio.dataio.read_audio(wav)
            if self.process_audio is not None:
                sig = self.process_audio(sig)
            if self.sample_mode == SampleMode.FULL:
                char = " ".join(wrd).upper()
                yield wrd_count
                yield sig
                yield 0.
                yield 1.
                yield wrd_start
                yield wrd_end
                yield phn_start
                yield phn_end
                yield wrd
                yield char
                yield 0.
                yield 1.
                yield phn
                yield 0.
                yield 1.
                yield char
                yield phn
                for value in args:
                    yield value
            else:
                yield self.sample_word_counts[idx].item()
                sample_start_idx = self.sample_start_idx[idx]
                sample_end_idx = self.sample_end_idx[idx]
                sig_len = sig.size(0)
                sig = sig[sample_start_idx:sample_end_idx]
                sig_rel_offset = sample_start_idx / sig_len
                sig_rel_length = (sample_end_idx - sample_start_idx) / sig_len
                yield sig
                yield sig_rel_offset
                yield sig_rel_length

                wrd_offset_start = self.wrd_offset_start[idx]
                wrd_offset_end = self.wrd_offset_end[idx]
                # wrd_start
                result = cut_offsets(wrd_start, wrd_offset_start, wrd_offset_end)
                yield result

                # wrd_end
                yield cut_offsets(wrd_end, wrd_offset_start, wrd_offset_end)
                # phn_start
                phn_start, phn_from, phn_to = cut_offsets_rel(
                    phn_start, wrd_start, wrd_offset_start, wrd_offset_end
                )
                yield phn_start
                # phn_end
                phn_end, _, _ = cut_offsets_rel(
                    phn_end, wrd_end, wrd_offset_start, wrd_offset_end
                )
                yield phn_end
                # wrd
                # TODO: Pre-compute this
                wrd_sample = wrd[wrd_offset_start:wrd_offset_end]
                char = " ".join(wrd_sample).upper()
                yield wrd_sample
                yield char
                snt = " ".join(wrd)
                char_abs_offset = len(" ".join(wrd[:wrd_offset_start]))
                if wrd_offset_start > 0:
                    char_abs_offset += 1
                char_rel_offset = char_abs_offset / len(snt)
                char_rel_length = len(char) / len(snt)
                yield char_rel_offset
                yield char_rel_length
                phn_cut, phn_rel_offset, phn_rel_length = cut_seq(phn, phn_from, phn_to)
                yield phn_cut
                yield phn_rel_offset
                yield phn_rel_length
                yield snt
                yield phn
                args_cut = len(self.audio_keys)
                audio_values, passthrough_values = args[:args_cut], args[args_cut:]
                audio_values_cut = [
                    cut_seq_rel(seq, sig_rel_offset, sig_rel_length)
                    for seq in audio_values
                ]
                for value in itertools.chain(audio_values_cut, passthrough_values):
                    yield value

        self.pipeline.add_dynamic_item(cut_sample)


def sample(base_dataset, num_samples, generator=None):
    """Retrieves a sample of the base dataset
    
    Arguments
    ---------
    base_dataset: DynamicItemDataset
        a base dataset
    num_samples: int
        the number of samples to include
    generator: torch.Generator
        a random number generator (optional)
        
    Returns
    -------
    dataset: FilteredSortedDynamicItemDataset
        a random sample of the dataset    
    """
    dataset = base_dataset
    if num_samples is not None and num_samples != len(base_dataset):
        sample_indexes = torch.multinomial(
            torch.ones(len(dataset)) / len(dataset),
            num_samples=num_samples,
            replacement=num_samples > len(base_dataset),
            generator=generator,
        )
        sample_data_ids = [dataset.data_ids[idx] for idx in sample_indexes]

        dataset = FilteredSortedDynamicItemDataset(
            from_dataset=dataset, data_ids=sample_data_ids
        )

    return dataset


PIPELINE_WRAPPER_ATTRS = {"pipeline", "key_map"}


class PipelineWrapper:
    """A pipeline wrapper that makes it possible to replace
    static outputs with dynamic ones. The trick is to output an
    item with the desired key prefixed with a '_'. The '_' will
    be removed in the output


    Arguments
    ---------
    pipeline: torch.tensor
        the original pipeline
    replace_keys: enumerable
        the list of keys that will be replaced

    """

    def __init__(self, pipeline, replace_keys):
        self.pipeline = pipeline
        self.key_map = {key: f"_{key}" for key in replace_keys}

    def compute_outputs(self, data):
        """Computes the output

        Arguments
        ---------
        data: dict
            the static data

        Returns
        -------
        result: dict
            the pipeline output
        """
        result = self.pipeline.compute_outputs(data)
        for key, key_r in self.key_map.items():
            if key_r in result:
                result[key] = result[key_r]
                del result[key_r]
        return result

    def set_output_keys(self, keys):
        """Sets the keys to be output by the pipeline

        Arguments
        ---------
        keys: enumerable
            a list of keys
        """
        keys_r = {self.key_map.get(key, key) for key in keys}
        self.pipeline.set_output_keys(keys_r)

    def __getattr__(self, name):
        """Delegates attribute calls to the underlying pipeline
        
        Arguments
        ---------
        name: str
            the attribute name

        Returns
        -------
        value: object
            the attribute value
        """
        if name in PIPELINE_WRAPPER_ATTRS:
            if name not in self.__dict__:
                raise AttributeError()
            return self.__dict__[name]

        return getattr(self.pipeline, name)


def time_to_index(times, sample_rate):
    """Converts a collection of time values to a list of
    wave array indexes at the specified sample rate

    Arguments
    ---------
    times: enumerable
        a list of time values
    sample_rate: int
        the sample rate (in hertz)

    Returns
    -------
    result: list
        a collection of indexes
    """

    if not torch.is_tensor(times):
        times = torch.tensor(times)

    return (times * sample_rate).floor().int().tolist()


def cut_offsets(offsets, start, end):
    """Given an array of offsets (e.g. word start times),
    returns a segment of it from <start> to <end> re-computed
    to begin at 0

    Arguments
    ---------
    offsets: list|torch.tensor
        a list or tensor of offsets

    start: int
        the starting index

    end: int
        the final index

    Returns
    -------
    result: list
        the re-calculated offset list
    """
    segment = offsets[start:end]
    if not torch.is_tensor(segment):
        segment = torch.tensor(segment)
    return (segment - segment[0]).tolist()


def cut_seq(seq, start, end):
    """Cuts a seuqence and computes the offset and length relative to the 
    total sequence length
    
    Arguments
    ---------
    seq: torch.Tensor|list-like
        the sequence
    start: int
        the start offset
    end: int
        the end offset
        
    Returns
    -------
    seq_cut: torch.Tensor|list-like
        the sub-sequence
    seq_rel_offset: float
        the relative offset
    seq_len_length: float
        the relative length"""
    seq_cut = seq[start:end]
    seq_len = len(seq)
    seq_rel_offset = start / seq_len
    seq_rel_length = (end - start) / seq_len
    return seq_cut, seq_rel_offset, seq_rel_length


def cut_seq_rel(seq, seq_rel_offset, seq_rel_length):
    """Cuts a sequence using pre-calculated relative offsets
    
    Arguments
    ---------
    seq : list | torch.Tensor
        the sequence to cut
    seq_rel_offset : float
        relative offset
    seq_rel_length : float
        relative length
    """
    seq_len = len(seq)
    start = int(seq_len * seq_rel_offset)
    end = int(start + seq_rel_length * seq_len)
    return seq[start:end]


def cut_offsets_rel(offsets, ref_offsets, start, end):
    """Given a sequence of offsets (e.g. phoneme offsets)
    and a reference sequence (e.g. sequence of words), finds
    the range in <offsets> corresponding to the specified range
    in <ref_offsets>

    Arguments
    ---------
    offsets: list|torch.Tensor
        a collection of offsets

    ref_offsets: list|torch.Tensor
        reference offsets

    Returns
    -------
    result: list
        the corresponding values in offsets
    start: int
        the start index
    end: int
        the end index
    """
    if not torch.is_tensor(offsets):
        offsets = torch.tensor(offsets)
    if not torch.is_tensor(ref_offsets):
        ref_offsets = torch.tensor(ref_offsets)
    start_value = ref_offsets[start].item()
    end_value = ref_offsets[end].item() if end < len(ref_offsets) else torch.inf
    condition = (offsets >= start_value) & (offsets < end_value)
    result = offsets[condition]
    result -= result[0].item()
    idx = condition.nonzero()

    return result.tolist(), idx.min().item(), idx.max().item() + 1


@checkpoints.register_checkpoint_hooks
class CurriculumController:
    """Provides control for the curriculum dataset from the training
    process"""

    def __init__(self, seed=42):
        self.dataset = None
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def bind(self, dataset):
        """Binds this controller to a dataset
        
        Arguments
        ---------
        dataset: CurriculumSpeechDataset
            a curriculum dataset
        """
        self.dataset = dataset

    def resample(
            self,
            min_words=None,
            max_words=None,
            num_samples=None,
            sample_mode=None,
            sorting=None,
        ):
        """Resamples the dataset
        
        Arguments
        ---------
        min_words: int
            the minimum number of words. If omitted, the value is not changed
        
        max_words: int
            the maximum number of words. If omitted, the value is not changed
        
        num_samples: int
            the number of samples. If omitted, the value is not changed
        
        sample_mode: SampleMode|str
            the sample mode to be used (full or segment)

        sorting: Sorting|str
            sorting mode
        """
        if self.dataset is None:
            raise ValueError("The curriculum controller is unbound")
        if min_words is None:
            min_words = self.dataset.min_words
        if max_words is None:
            max_words = self.dataset.max_words
        if num_samples is None:
            num_samples = self.dataset.num_samples
        if sample_mode is None:
            sample_mode = SampleMode.SEGMENT
        else:
            sample_mode = SampleMode(sample_mode)
        
        if sorting is None:
            sorting = Sorting.RANDOM
        else:
            sorting = Sorting(sorting)

        self.dataset.min_words = min_words
        self.dataset.max_words = max_words
        self.dataset.num_samples = num_samples
        self.dataset.sample_mode = sample_mode
        self.dataset.sorting = sorting
        self.dataset.sample_segments(generator=self.generator)

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        if self.generator is not None:
            data = {"generator": self.generator.get_state()}
        else:
            data = {}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device  # Unused in here
        state = torch.load(path)
        generator_state = state.get("generator")
        if generator_state is not None:
            self.generator.set_state(generator_state)

SAMPLING_OPTIONS = ["min_words", "max_words", "num_samples", "sample_mode", "sorting"]

@checkpoints.register_checkpoint_hooks
class Curriculum:
    """A helper class to define a curriculum
    
    Arguments
    ---------
    steps: list
        a list of dicts similar to the following
        [
            {"epoch": 1, "max_words": 3},
            {"epoch": 5, "max_words": 5},
            {"epoch": 10, "max_words": 5},
        ]
    controller: CurriculumController
        a curriculum controller
    """

    def __init__(self, steps, controller=None):
        self.steps = sorted(
            [{"epoch": 0, **step} for step in steps],
            key=lambda step: step["epoch"],
        )
        if controller is None:
            controller = CurriculumController()
        self.controller = controller
        self.step_id = None

    def apply(self, epoch):
        """Finds the step corresponding to the specified
        epoch
        
        Arguments
        ---------
        epoch: int
            the epoch number
            
        Returns
        -------
        step_id: int
            the step ID / number (starting at 1)
        step: dict
            the step configuration
        """
        step_id, step = self.find_step(epoch)
        if step_id is None:
            logger.warn("Unable to find a curriculum step epoch %d", epoch)
            return None, None
        self.apply_step(step_id)
        return step_id, step
    
    def apply_step(self, step_id):
        """Applies the specified curriculum step
        
        Arguments
        ---------
        step_id: int
            the step number (starting at 1)
            
        Returns
        -------
        step: dict
            the step definition
        """
        step = self.steps[step_id - 1]
        kwargs = {key: step.get(key) for key in SAMPLING_OPTIONS}
        self.controller.resample(**kwargs)
        self.step_id = step_id
        return step

    def debug(self):
        """Puts the curriculum in debug mode - useful to test if the specified
        curriculum fits in the available memory"""
        for idx, step in enumerate(self.steps, start=1):
            step["epoch"] = idx     
            # NOTE: Make sure the biggest sequence fits
            step["sorting"] = Sorting.DESCENDING       

    def find_step(self, epoch):
        """Finds the step corresponding to the specified
        epoch
        
        Arguments
        ---------
        epoch: int
            the epoch number
            
        Returns
        -------
        step_id: int
            the step ID / number (starting at 1)
        step: dict
            the step configuration
        """
        return next(
            (
                (step_id, step)
                for step_id, step in reversed(
                    list(enumerate(self.steps, start=1))
                )
                if epoch >= step["epoch"]
            ),
            (None, None),
        )
    
    def is_end_of_step(self, epoch):
        """A convenience way to determine whether or not the specified
        epoch is at the end of the curriculum step.

        This is particularly useful if the state of training at a particular
        curriculum step needs to be retained
        
        Arguments
        ---------
        epoch: int
            the epoch number
            
        Returns
        -------
        result: bool
            whether it is the end of the epoch"""
        next_epoch = epoch + 1
        _, step = self.find_step(next_epoch)
        return step["epoch"] == next_epoch
    
    def epoch_within_step(self, epoch):
        """Returns the epoch number relative to the beginning of the step,
        starting at 1
        
        Arguments
        ---------
        epoch: int
            the absolute epoch number
        
        Returns
        -------
        relative_epoch: int
            the epoch relative to the start of the step"""
        
        _, step = self.find_step(epoch)
        step_start = step.get("epoch", 1)
        return epoch - step_start + 1

    def bind(self, dataset):
        """Binds the underlying controller to a dataset
        
        Arguments
        ---------
        dataset: CurriculumSpeechDataset
            a curriculum dataset
        """
        self.controller.bind(dataset)

    def save_dataset(self, path=None, keys=None):
        """Saves the dataset contents for future reference, analysis
        and debugging
        
        Arguments
        ---------
        dataset_key: str
            a string key to identify the dataset
        path: str
            the filesystem path
        keys: list
            the data keys to output
        """
        dataset = self.controller.dataset
        if path is None:
            path = "."
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, f"data-{self.step_id}.json")
        if not os.path.exists(file_name):
            if keys is not None:
                with dataset.output_keys_as(keys):
                    dataset.to_json(file_name)
            else:
                dataset.to_json(file_name)

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        self.controller.save(path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        self.controller.load(path, end_of_epoch, device)
