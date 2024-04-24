from .core.asr import *
from .core.lm import *

run_opts["device"] = "cuda"
asr_brain = ASR(
    modules=hparams["modules"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
)

# Adding objects to trainer.
asr_brain.tokenizer = label_encoder
asr_brain.checkpointer.recover_if_possible(device="cuda")
asr_brain.modules.eval()