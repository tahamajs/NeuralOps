"""SparseGPT: One‑shot pruning via second‑order information. Registered as 'sparsegpt'."""

import subprocess
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("sparsegpt")
class SparseGPTCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.5, device="cuda", nsamples=128, **kwargs):
        super().__init__(model, name="sparsegpt", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.nsamples = nsamples

    def compress(self, save_path="pruned_model"):
        temp_path = Path("temp_model_for_sparsegpt")
        temp_path.mkdir(exist_ok=True)
        self.model.save_pretrained(temp_path)
        self.tokenizer.save_pretrained(temp_path)

        cmd = [
            "python", "-m", "sparsegpt.opt",
            str(temp_path),
            "c4",
            "--sparsity", str(self.sparsity),
            "--nsamples", str(self.nsamples),
            "--save", save_path
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SparseGPT failed: {e}")

        pruned_model = AutoModelForSequenceClassification.from_pretrained(save_path)
        pruned_tokenizer = AutoTokenizer.from_pretrained(save_path)
        shutil.rmtree(temp_path, ignore_errors=True)
        self.model = pruned_model   # update base model
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()