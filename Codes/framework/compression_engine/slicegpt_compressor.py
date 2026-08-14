"""SliceGPT PCA‑based slicing compressor. Registered as 'slicegpt_pca'."""
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from sklearn.decomposition import PCA


class SliceGPTPCACompressor(BaseCompressor):
    """
    SliceGPT PCA-based compressor that reduces input dimensions of Linear layers.
    
    This compressor keeps only the top principal components of a reference layer's
    output activations. It uses soft pruning (zeroing out columns) to maintain
    model compatibility without altering weight shapes.

    Attributes:
        calibration_loader (Any): DataLoader providing calibration data.
        sparsity (float): Fraction of variance to discard (compression ratio).
        tokenizer (Any): Tokenizer for generating calibration data if loader is None.
        calibration_texts (List[str]): Texts used for calibration.
    """

    def __init__(
        self,
        model: Any,
        calibration_loader: Optional[Any] = None,
        sparsity: float = 0.3,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes the SliceGPTPCACompressor.

        Args:
            model (Any): The PyTorch model to compress.
            calibration_loader (Optional[Any], optional): DataLoader for calibration data. Defaults to None.
            sparsity (float, optional): Sparsity ratio (fraction of PCA components to prune). Defaults to 0.3.
            tokenizer (Optional[Any], optional): Tokenizer to process calibration_texts. Defaults to None.
            calibration_texts (Optional[List[str]], optional): List of sentences for calibration if no loader is provided. Defaults to None.
            **kwargs: Additional arguments for BaseCompressor.
        """
        super().__init__(model, name="slicegpt_pca", **kwargs)
        self.calibration_loader = calibration_loader
        self.sparsity = sparsity
        self.tokenizer = tokenizer
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subfield of artificial intelligence.",
            "Transformers have revolutionized natural language processing.",
        ]

    def compress(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Compresses the model using PCA-based soft pruning on Linear layers.

        It hooks into a reference layer to gather activations on calibration data,
        computes PCA to find the most significant components, and then zeroes out
        the weight columns of all Linear layers that correspond to the least
        significant components based on the sparsity target.

        Returns:
            Tuple[Any, Dict[str, Any]]: A tuple containing the compressed model and statistics.
                - model: The compressed PyTorch model.
                - stats: Compression statistics dictionary.

        Raises:
            RuntimeError: If a forward hook cannot be attached.
            ValueError: If neither a calibration_loader nor a tokenizer is provided.
        """
        activations = []
        handle = None
        # Prefer the first encoder layer; fallback to first Linear
        for name, module in self.model.named_modules():
            if "encoder.layer.0" in name:
                handle = module.register_forward_hook(
                    lambda m, i, o: activations.append(o.detach().clone().cpu().numpy())
                )
                break
        if handle is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    handle = module.register_forward_hook(
                        lambda m, i, o: activations.append(o.detach().clone().cpu().numpy())
                    )
                    break
        if handle is None:
            raise RuntimeError("Could not attach hook for SliceGPT PCA")

        # 2. Run calibration forward pass
        device = next(self.model.parameters()).device
        if self.calibration_loader is not None:
            with torch.no_grad():
                for batch in self.calibration_loader:
                    batch = {
                        k: v.to(device)
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
                k: v.to(device) for k, v in inputs.items()
            }
            with torch.no_grad():
                _ = self.model(**inputs)

        handle.remove()

        # 3. Perform PCA
        activations_arr = np.concatenate(activations, axis=0)
        activations_arr = activations_arr.reshape(-1, activations_arr.shape[-1])
        pca = PCA(n_components=activations_arr.shape[-1])
        pca.fit(activations_arr)
        
        # Add epsilon guard for stability in variance ratio calculations (just in case)
        explained_variance_ratio_ = pca.explained_variance_ratio_ + 1e-8
        explained_variance_ratio_ = explained_variance_ratio_ / explained_variance_ratio_.sum()
        cumsum = np.cumsum(explained_variance_ratio_)
        
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
def slicegpt_pca_factory(model: Any, comp_config: Any) -> Any:
    """
    Factory function to instantiate and execute the SliceGPT PCA compressor.

    Args:
        model (Any): The PyTorch model to compress.
        comp_config (Any): Configuration object containing compression parameters.
            Can include `slicegpt_pca_sparsity` and `slicegpt_pca_calib_texts`.

    Returns:
        Any: The compressed model.
    """
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
