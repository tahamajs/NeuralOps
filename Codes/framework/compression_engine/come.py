"""CoMe: Compress LLMs through layer concatenation (external wrapper). Registered as 'come'."""

import subprocess
import shutil
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class CoMeCompressor(BaseCompressor):
    """
    Applies CoMe layer concatenation to the model.
    Requires the official CoMe repository to be installed and accessible
    via `python -m main`.

    Config parameters (in comp_config):
        come_target_layers : int, optional – number of layers after merging
        come_sparsity       : float, default 0.3 – fraction of layers to remove
        come_calib_dataset : str, default 'wiki2'
        come_nsamples      : int, default 256
    """

    def __init__(self, model, tokenizer, target_layers=None, sparsity=0.3,
                 device="cuda", **kwargs):
        super().__init__(model, name="come", **kwargs)
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.sparsity = sparsity
        self.device = device

    def compress(self, save_dir="come_pruned", calibration_dataset="wiki2",
                 nsamples=256):
        # 1. Save temporary model
        temp_path = Path("temp_come_model")
        temp_path.mkdir(exist_ok=True)
        self.model.save_pretrained(temp_path)
        self.tokenizer.save_pretrained(temp_path)

        # 2. Determine target layers
        if self.target_layers is None:
            orig_layers = self.model.config.num_hidden_layers
            self.target_layers = int(orig_layers * (1 - self.sparsity))

        # 3. Run CoMe script
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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"CoMe failed: {e.stderr}\nStdout: {e.stdout}"
            )

        # 4. Load compressed model
        compressed_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        compressed_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        # 5. Clean up
        shutil.rmtree(temp_path, ignore_errors=True)

        # Update internal model
        self.model = compressed_model
        # Attach tokenizer for downstream use
        self.model.tokenizer = compressed_tokenizer

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("come")
def come_factory(model, comp_config):
    target_layers = getattr(comp_config, 'come_target_layers', None)
    sparsity = getattr(comp_config, 'come_sparsity', 0.3)
    calib_dataset = getattr(comp_config, 'come_calib_dataset', 'wiki2')
    nsamples = getattr(comp_config, 'come_nsamples', 256)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("CoMe compressor requires tokenizer attached as model.tokenizer")

    compressor = CoMeCompressor(
        model, tokenizer,
        target_layers=target_layers,
        sparsity=sparsity
    )
    return compressor.compress(
        calibration_dataset=calib_dataset,
        nsamples=nsamples
    )[0]   # return only the model