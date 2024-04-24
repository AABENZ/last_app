from pyctcdecode import build_ctcdecoder
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

hparams_file, run_opts, overrides = sb.parse_arguments(["model/train_semi.yaml"])

# If distributed_launch=True then
# create ddp_group with the right communication protocol
# sb.utils.distributed.ddp_init_group(run_opts)

with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

label_encoder = sb.dataio.encoder.CTCTextEncoder()

lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
special_labels = {
    "blank_label": hparams["blank_index"],
    "unk_label": hparams["unk_index"]
}
label_encoder.load_or_create(
    path=lab_enc_file,
    from_didatasets=[[]],
    output_key="char_list",
    special_labels=special_labels,
    sequence_input=True,
)

ind2lab = label_encoder.ind2lab
# print("ind2lab ===============================")
# print(ind2lab)
# print("ind2lab ===============================")
labels = [ind2lab[x] for x in range(len(ind2lab))]
labels = [""] + labels[1:-1] + ["1"]
# Replace the <blank> token with a blank character, needed for PyCTCdecode
# print("labels:====================================")
# print(labels)
# print("labels:====================================")
decoder = build_ctcdecoder(
    labels,
    kenlm_model_path="model/outdomain.arpa",  # .arpa or .bin
    alpha=0.5,  # Default by KenLM
    beta=1.0,  # Default by KenLM
)
