"""
SliceGPT: Post‑training sparsification that reduces embedding dimension.
Paper: https://arxiv.org/abs/2401.15024
Code: https://github.com/microsoft/TransformerCompression
"""
import subprocess
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from framework.core.base import BaseCompressor      # note: from compression.core.base

class SliceGPTCompressor(BaseCompressor):   # note the inheritance
    def __init__(self, model, tokenizer, sparsity=0.25, device="cuda", use_wandb=False):
        self.model = model
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.use_wandb = use_wandb

    def compress(self, save_dir="sliced_model", run_finetuning=False, finetune_nsamples=8000):
        temp_path = Path("temp_original_model")
        temp_path.mkdir(exist_ok=True)
        self.model.save_pretrained(temp_path)
        self.tokenizer.save_pretrained(temp_path)

        cmd = [
            "python", "-m", "slicegpt",
            "--model", str(temp_path),
            "--save-dir", save_dir,
            "--sparsity", str(self.sparsity),
            "--device", self.device
        ]
        if not self.use_wandb:
            cmd.append("--no-wandb")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SliceGPT failed: {e}")

        sliced_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        sliced_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        if run_finetuning:
            finetune_cmd = [
                "python", "-m", "slicegpt.finetune",
                "--model", str(temp_path),
                "--sliced-model-path", save_dir,
                "--sparsity", str(self.sparsity),
                "--device", self.device,
                "--finetune-train-nsamples", str(finetune_nsamples)
            ]
            subprocess.run(finetune_cmd, check=True)
            sliced_model = AutoModelForSequenceClassification.from_pretrained(save_dir)

        shutil.rmtree(temp_path, ignore_errors=True)
        return sliced_model, sliced_tokenizer
    



@COMPRESSOR_REGISTRY.register("slicegpt")
def slicegpt_factory(model, comp_config):
    sparsity = getattr(comp_config, 'slicegpt_sparsity', 0.25)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("slicegpt requires tokenizer attached as model.tokenizer")
    compressor = SliceGPTCompressor(model, tokenizer, sparsity)
    compressed_model, _ = compressor.compress()
    return compressed_model

