import subprocess
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CoMeCompressor:
    """
    CoMe: Compress LLMs through layer concatenation.
    Paper: https://arxiv.org/abs/2510.15304
    Code: https://github.com/MPI-Lab/CoMe
    """

    def __init__(self, model, tokenizer, target_layers=None, sparsity=0.3, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers          # number of layers to keep after merging
        self.sparsity = sparsity
        self.device = device

    def compress(self, save_dir="come_pruned", calibration_dataset="wiki2", nsamples=256):
        """
        Apply CoMe pruning (layer concatenation) and return the compressed model + tokenizer.
        """
        # --------------------------------------------------------------------
        # 1. Save the original model and tokenizer to a temporary directory
        # --------------------------------------------------------------------
        temp_path = Path("temp_come_model")
        temp_path.mkdir(exist_ok=True)
        self.model.save_pretrained(temp_path)
        self.tokenizer.save_pretrained(temp_path)

        # --------------------------------------------------------------------
        # 2. Determine the target number of layers if not provided
        # --------------------------------------------------------------------
        if self.target_layers is None:
            orig_layers = self.model.config.num_hidden_layers
            self.target_layers = int(orig_layers * (1 - self.sparsity))

        # --------------------------------------------------------------------
        # 3. Run the CoMe script (assumes the CoMe repository is installed and accessible)
        # --------------------------------------------------------------------
        cmd = [
            "python", "-m", "main",
            "--method", "concat_merge",
            "--model-name", str(temp_path),
            "--target-layers", str(self.target_layers),
            "--save-path", save_dir,
            "--calibration-dataset", calibration_dataset,
            "--skip-method", "bi",
            "--nsamples", str(nsamples),
            "--wo-repeat"
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"CoMe failed: {e.stderr}\nStdout: {e.stdout if hasattr(e, 'stdout') else ''}"
            )

        # --------------------------------------------------------------------
        # 4. Load the pruned model
        # --------------------------------------------------------------------
        compressed_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        compressed_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        # --------------------------------------------------------------------
        # 5. Clean up temporary files
        # --------------------------------------------------------------------
        shutil.rmtree(temp_path, ignore_errors=True)

        return compressed_model, compressed_tokenizer




