#!/usr/bin/env/python3
"""Recipe for training a Text-to-Speech system based on tokenized audio - Encodec version

Inspired by WhisperSpeech
https://github.com/collabora/WhisperSpeech

However, this is not an implementation of WhisperSpeech, but rather
a radical simplification of it that uses only an acoustic model


Authors
 * Artem Ploujnikov 2024
"""

from train import TokotronBrain, run_experiment
from speechbrain.dataio.dataio import clean_padding_


class TokotronEncodecBrain(TokotronBrain):
    """Tokotron implementation for Encodec"""

    def create_waveform(self, audio, length, emb):
        """Creates a waveform from a discrete or continuous audio
        representation

        Arguments
        ---------
        audio : torch.Tensor
            An audio tensor (Batch x Length x Heads or Batch x Length x Heads x Features)
        lengths : torch.Tensor
            A 1-D tensor
        emb: dict
            Embeddings (speaker, etc)

        Returns
        -------
        wav : torch.Tensor
        """
        wav = self.modules.tokenizer.decode(audio)
        wav = wav.squeeze(1)
        clean_padding_(wav, length)
        return wav


if __name__ == "__main__":
    run_experiment(TokotronEncodecBrain)
