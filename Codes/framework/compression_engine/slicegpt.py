"""SliceGPT: post‑training slicing (external wrapper). Registered as 'slicegpt'."""
from typing import Any, Optional, Tuple, Dict

import shutil
import subprocess
from pathlib import Path

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SliceGPTCompressor(BaseCompressor):
    """
    Wraps the official SliceGPT tool for post-training slicing.
    
    This class handles the execution of the external `slicegpt` module to perform
    network slicing (pruning) and optional fine-tuning.

    Attributes:
        model: The model to be compressed.
        tokenizer: The tokenizer associated with the model.
        sparsity (float): The target sparsity for compression (e.g., 0.25).
        device (str): The device to use for computation ("cuda" or "cpu").
        use_wandb (bool): Whether to log metrics to Weights & Biases.
    """

    def __init__(
        self, model: Any, tokenizer: Any, sparsity: float = 0.25, device: str = "cuda", use_wandb: bool = False, **kwargs: Any
    ) -> None:
        """
        Initializes the SliceGPTCompressor.

        Args:
            model (Any): The PyTorch model to compress.
            tokenizer (Any): The tokenizer for the model.
            sparsity (float, optional): Target sparsity ratio (compression ratio). Defaults to 0.25.
            device (str, optional): Target device for operations. Defaults to "cuda".
            use_wandb (bool, optional): If True, uses W&B for logging. Defaults to False.
            **kwargs: Additional keyword arguments passed to BaseCompressor.
        """
        super().__init__(model, name="slicegpt", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.use_wandb = use_wandb

    def compress(
        self, save_dir: str = "sliced_model", run_finetuning: bool = False, finetune_nsamples: int = 8000
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the SliceGPT compression and optional fine-tuning.

        This method saves the model temporarily, invokes the `slicegpt` module via
        a subprocess, and then loads the sliced model back.

        Args:
            save_dir (str, optional): Directory to save the compressed model. Defaults to "sliced_model".
            run_finetuning (bool, optional): Whether to run fine-tuning after slicing. Defaults to False.
            finetune_nsamples (int, optional): Number of calibration samples for fine-tuning. Defaults to 8000.

        Returns:
            Tuple[Any, Dict[str, Any]]: A tuple containing the compressed model and its compression statistics.
        """
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
        sliced_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(self.device)
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
            sliced_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(self.device)

        # 5. Clean up
        shutil.rmtree(temp_path, ignore_errors=True)

        # Update internal model
        self.model = sliced_model
        self.model.tokenizer = sliced_tokenizer

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("slicegpt")
def slicegpt_factory(model: Any, comp_config: Any) -> Any:
    """
    Factory function to instantiate and run the SliceGPT compressor.

    Args:
        model (Any): The PyTorch model to be compressed.
        comp_config (Any): Configuration object containing compression parameters.
            Must have `slicegpt_sparsity`, `slicegpt_run_finetuning`, and `slicegpt_finetune_nsamples` attributes.

    Returns:
        Any: The compressed model.
        
    Raises:
        ValueError: If the model does not have a `tokenizer` attached.
    """
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
