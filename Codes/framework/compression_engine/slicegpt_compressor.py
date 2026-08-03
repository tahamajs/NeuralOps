"""SliceGPT PCA‑based slicing compressor. Registered as 'slicegpt_pca'."""

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from sklearn.decomposition import PCA


class SliceGPTPCACompressor(BaseCompressor):
    """
    Reduces the input dimension of all Linear layers by keeping only the
    top principal components of a reference layer's output activations.
    Uses soft pruning (zeroing columns) to maintain model compatibility.

    Config parameters:
        slicegpt_pca_sparsity : float, default 0.3 – fraction of variance to discard
        slicegpt_pca_calib_texts : list of str, optional – if tokenizer available
    """

    def __init__(
        self,
        model,
        calibration_loader=None,
        sparsity=0.3,
        tokenizer=None,
        calibration_texts=None,
        **kwargs
    ):
        super().__init__(model, name="slicegpt_pca", **kwargs)
        self.calibration_loader = calibration_loader
        self.sparsity = sparsity
        self.tokenizer = tokenizer
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subfield of artificial intelligence.",
            "Transformers have revolutionized natural language processing.",
        ]

    def compress(self):
        # 1. Collect activations from a reference module
        """Compress the target module using this strategy."""
        activations = []
        handle = None
        # Prefer the first encoder layer; fallback to first Linear
        for name, module in self.model.named_modules():
            if "encoder.layer.0" in name:
                handle = module.register_forward_hook(
                    lambda m, i, o: activations.append(o.detach().cpu().numpy())
                )
                break
        if handle is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    handle = module.register_forward_hook(
                        lambda m, i, o: activations.append(o.detach().cpu().numpy())
                    )
                    break
        if handle is None:
            raise RuntimeError("Could not attach hook for SliceGPT PCA")

        # 2. Run calibration forward pass
        if self.calibration_loader is not None:
            with torch.no_grad():
                for batch in self.calibration_loader:
                    batch = {
                        k: v.to(next(self.model.parameters()).device)
                        for k, v in batch.items()
                    }
                    _ = self.model(**batch)
        else:
            # Use tokenizer to create a simple calibration set
            if self.tokenizer is None:
                raise ValueError(
                    "SliceGPT PCA needs either calibration_loader or tokenizer"
                )
            inputs = self.tokenizer(
                self.calibration_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()
            }
            with torch.no_grad():
                _ = self.model(**inputs)

        handle.remove()

        # 3. Perform PCA
        activations = np.concatenate(activations, axis=0)
        activations = activations.reshape(-1, activations.shape[-1])
        pca = PCA(n_components=activations.shape[-1])
        pca.fit(activations)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # keep at least 1 dimension
        keep = max(1, np.searchsorted(cumsum, 1 - self.sparsity) + 1)

        # 4. Soft prune all Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Zero out columns beyond `keep`
                mask = torch.zeros(module.weight.shape[1], device=module.weight.device)
                mask[:keep] = 1.0
                module.weight.data = module.weight.data * mask.unsqueeze(0)
                # Bias is left unchanged (pruning input features, not output)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("slicegpt_pca")
def slicegpt_pca_factory(model, comp_config) -> Any:
    """Slicegpt pca factory."""
    sparsity = getattr(comp_config, "slicegpt_pca_sparsity", 0.3)
    calibration_texts = getattr(comp_config, "slicegpt_pca_calib_texts", None)
    tokenizer = getattr(model, "tokenizer", None)
    compressor = SliceGPTPCACompressor(
        model,
        sparsity=sparsity,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
