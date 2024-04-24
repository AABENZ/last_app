#!/usr/bin/env python3
import sys
import torch
import logging
# import gradio as gr
import speechbrain as sb
from pathlib import Path
import os
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from .lm import decoder

logger = logging.getLogger(__name__)


from speechbrain.pretrained import EncoderASR


# Define training procedure
class ASR(sb.core.Brain):

    def treat_wav(self,sig):
        feats = self.modules.wav2vec2(sig.to("cuda"), torch.tensor([1]).to("cuda"))
        feats = self.modules.enc(feats)
        logits = self.modules.ctc_lin(feats)
        p_ctc = self.hparams.log_softmax(logits)
        predicted_words =[]
        for logs in p_ctc:
            text = decoder.decode(logs.detach().cpu().numpy())
            predicted_words.append(text.split(" "))
        return " ".join(predicted_words[0])


    # def treat_wav(self,sig,batch_size):
    #     feats = self.modules.wav2vec2(sig.to("cuda"), torch.ones(batch_size).to("cuda"))
    #     feats = self.modules.enc(feats)
    #     logits = self.modules.ctc_lin(feats)
    #     p_ctc = self.hparams.log_softmax(logits)
    #     predicted_words =[]
    #     for logs in p_ctc:
    #         text = decoder.decode(logs.detach().cpu().numpy())
    #         predicted_words.append(text.split(" "))
    #     return " ".join(p_ctc)
    
    def encode_batch(self, wavs, wav_lens):
            """Encodes the input audio into a sequence of hidden states

            The waveforms should already be in the model's desired format.
            You can call:
            ``normalized = EncoderASR.normalizer(signal, sample_rate)``
            to get a correctly converted signal in most cases.

            Arguments
            ---------
            wavs : torch.Tensor
                Batch of waveforms [batch, time, channels] or [batch, time]
                depending on the model.
            wav_lens : torch.Tensor
                Lengths of the waveforms relative to the longest one in the
                batch, tensor of shape [batch]. The longest one should have
                relative length 1.0 and others len(waveform) / max_length.
                Used for ignoring padding.

            Returns
            -------
            torch.Tensor
                The encoded batch
            """
            wavs = wavs.float()
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            encoder_out = self.encoder(wavs, wav_lens)
            return encoder_out

    
# print(ASR.__dict__)