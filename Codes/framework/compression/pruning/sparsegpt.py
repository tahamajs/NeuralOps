"""
SparseGPT: One‑shot pruning using second‑order information.
Paper: https://arxiv.org/abs/2301.00774
Code: https://github.com/IST-DASLab/sparsegpt
"""
import subprocess
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SparseGPTPruner:
    def __init__(self, model, tokenizer, sparsity=0.5, device="cuda", nsamples=128):
        self.model = model
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.nsamples = nsamples

    def prune(self, save_path="pruned_model"):
        temp_path = Path("temp_model_for_sparsegpt")
        temp_path.mkdir(exist_ok=True)
        self.model.save_pretrained(temp_path)
        self.tokenizer.save_pretrained(temp_path)

        # Adapt as needed – this example uses the OPT script
        cmd = [
            "python", "-m", "sparsegpt.opt",
            str(temp_path),
            "c4",                           # calibration dataset
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
        return pruned_model, pruned_tokenizer