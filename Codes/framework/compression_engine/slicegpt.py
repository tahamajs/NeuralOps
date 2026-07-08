"""SliceGPT: post‑training slicing (external wrapper). Registered as 'slicegpt'."""

import shutil
import subprocess
from pathlib import Path

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SliceGPTCompressor(BaseCompressor):
    """
    Wraps the official SliceGPT tool.
    Requires `slicegpt` package installed (`pip install slicegpt`).

    Config parameters:
        slicegpt_sparsity       : float, default 0.25
        slicegpt_run_finetuning : bool, default False
        slicegpt_finetune_nsamples : int, default 8000
    """

    def __init__(
        self, model, tokenizer, sparsity=0.25, device="cuda", use_wandb=False, **kwargs
    ):
        super().__init__(model, name="slicegpt", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.use_wandb = use_wandb

    def compress(
        self, save_dir="sliced_model", run_finetuning=False, finetune_nsamples=8000
    ):
        # 1. Save temporary model
        temp_path = Path("temp_original_model")
        temp_path.mkdir(exist_ok=True)
        self.model.save_pretrained(temp_path)
        self.tokenizer.save_pretrained(temp_path)

        # 2. Run slicing
        cmd = [
            "python",
            "-m",
            "slicegpt",
            "--model",
            str(temp_path),
            "--save-dir",
            save_dir,
            "--sparsity",
            str(self.sparsity),
            "--device",
            self.device,
        ]
        if not self.use_wandb:
            cmd.append("--no-wandb")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SliceGPT failed: {e}")

        # 3. Load compressed model
        sliced_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        sliced_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        # 4. Optional fine‑tuning
        if run_finetuning:
            finetune_cmd = [
                "python",
                "-m",
                "slicegpt.finetune",
                "--model",
                str(temp_path),
                "--sliced-model-path",
                save_dir,
                "--sparsity",
                str(self.sparsity),
                "--device",
                self.device,
                "--finetune-train-nsamples",
                str(finetune_nsamples),
            ]
            subprocess.run(finetune_cmd, check=True)
            sliced_model = AutoModelForSequenceClassification.from_pretrained(save_dir)

        # 5. Clean up
        shutil.rmtree(temp_path, ignore_errors=True)

        # Update internal model
        self.model = sliced_model
        self.model.tokenizer = sliced_tokenizer

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("slicegpt")
def slicegpt_factory(model, comp_config):
    sparsity = getattr(comp_config, "slicegpt_sparsity", 0.25)
    run_finetuning = getattr(comp_config, "slicegpt_run_finetuning", False)
    finetune_nsamples = getattr(comp_config, "slicegpt_finetune_nsamples", 8000)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("SliceGPT requires tokenizer attached as model.tokenizer")

    compressor = SliceGPTCompressor(model, tokenizer, sparsity=sparsity)
    return compressor.compress(
        run_finetuning=run_finetuning, finetune_nsamples=finetune_nsamples
    )[0]
