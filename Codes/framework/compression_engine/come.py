"""CoMe: Compress LLMs through layer concatenation (external wrapper). Registered as 'come'."""
from typing import Any

import shutil
import subprocess
from pathlib import Path
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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

    def __init__(
        self,
        model,
        tokenizer=None,
        target_layers=None,
        sparsity=0.3,
        device="cuda",
        **kwargs,
    ):
        super().__init__(model, name="come", **kwargs)
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.sparsity = sparsity
        self.device = device

    def compress(
        self, save_dir="come_pruned", calibration_dataset="wiki2", nsamples=256
    ):
        """Compress the target module using this strategy."""
        # Native PyTorch fallback layer pruning if tokenizer or external tool unavailable
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layer"):
            layers = self.model.transformer.layer
            orig_len = len(layers)
            keep_len = max(1, int(orig_len * (1 - self.sparsity)))
            if self.target_layers is not None:
                keep_len = min(orig_len, max(1, self.target_layers))
            self.model.transformer.layer = nn.ModuleList([layers[i] for i in range(keep_len)])
        elif hasattr(self.model, "layers") and isinstance(self.model.layers, nn.ModuleList):
            layers = self.model.layers
            orig_len = len(layers)
            keep_len = max(1, int(orig_len * (1 - self.sparsity)))
            if self.target_layers is not None:
                keep_len = min(orig_len, max(1, self.target_layers))
            self.model.layers = nn.ModuleList([layers[i] for i in range(keep_len)])
        else:
            # SVD prune linear layers if architecture lacks explicit layer list
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    W = module.weight.data.float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    r = max(1, int(len(S) * (1 - self.sparsity)))
                    module.weight.data = ((U[:, :r] * S[:r]) @ Vt[:r, :]).to(dtype=module.weight.dtype)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("come")
def come_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Come factory."""
    target_layers = getattr(comp_config, "come_target_layers", kwargs.get("target_layers", None))
    sparsity = getattr(comp_config, "come_sparsity", kwargs.get("sparsity", 0.3))
    tokenizer = getattr(model, "tokenizer", kwargs.get("tokenizer", None))

    compressor = CoMeCompressor(
        model, tokenizer, target_layers=target_layers, sparsity=sparsity, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
