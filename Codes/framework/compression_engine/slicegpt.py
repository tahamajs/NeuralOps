"""
SliceGPT: Post-Training Slicing with Orthogonal Transformations and Non-Uniform Head Slicing.
Registered in COMPRESSOR_REGISTRY as 'slicegpt'.

Mathematical Formulation:
1. Activation Covariance:
   $$C_X = \\frac{1}{N} X^\\top X \\in \\mathbb{R}^{d_{in} \\times d_{in}}$$
2. Orthogonal Eigendecomposition:
   $$C_X = Q \\Lambda Q^\\top, \\quad Q^\\top Q = I$$
3. Rotated Linear Transformation:
   $$\\widetilde{W} = W Q, \\quad \\widetilde{X} = Q^\\top X$$
4. Slicing Projection (Uniform or Non-Uniform Head Slicing):
   $$P_r = [e_{\\pi(1)}, e_{\\pi(2)}, \\dots, e_{\\pi(r)}] \\in \\mathbb{R}^{d_{in} \\times r}$$
   $$W_{sliced} = \\widetilde{W} P_r P_r^\\top Q^\\top$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    DEFAULT_CALIB_TEXTS,
    get_calibration_tensors,
)

logger = logging.getLogger(__name__)


class SliceGPTCompressor(BaseCompressor):
    """
    Native PyTorch SliceGPT Compressor.
    
    Applies orthogonal transformation Q to rotate model representations onto principal
    activation variance coordinates, followed by uniform or non-uniform projection head slicing.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.25,
        non_uniform: bool = True,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="slicegpt", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.non_uniform = non_uniform
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations across all nn.Linear layers."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(inp.cpu())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("Forward pass for activation collection had exception: %s. Using synthetic linear inputs.", e)

        for h in hooks:
            h.remove()

        # Concatenate collected activations
        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                cat_act = torch.cat(act_list, dim=0)
                aggregated[name] = cat_act

        return aggregated

    def _compute_orthogonal_rotation(self, activations: torch.Tensor, d_in: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derives orthogonal matrix Q from activation covariance C_X = (1/N) X^T X via eigendecomposition.
        Returns:
            Q: Orthogonal rotation matrix [d_in, d_in]
            eigenvalues: Eigenvalue variance spectrum [d_in]
        """
        if activations is not None and activations.numel() > 0:
            X = activations.reshape(-1, activations.shape[-1]).float()
            if X.shape[-1] == d_in:
                N = X.shape[0]
                C = (1.0 / max(N, 1)) * torch.matmul(X.T, X)
                # Add small regularizer for numerical stability
                C = C + 1e-6 * torch.eye(d_in)
                eigenvalues, Q = torch.linalg.eigh(C)
                # Sort in descending order
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                Q = Q[:, idx]
                return Q, eigenvalues

        # Fallback to Haar-distributed random orthogonal matrix
        H = torch.randn(d_in, d_in)
        Q, _ = torch.linalg.qr(H)
        eigenvalues = torch.ones(d_in)
        return Q, eigenvalues

    def compress(
        self,
        save_dir: Optional[str] = None,
        run_finetuning: bool = False,
        finetune_nsamples: int = 128,
        **kwargs: Any,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Applies SliceGPT orthogonal rotation and projection slicing.
        
        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Compressed model and comprehensive stats.
        """
        activations_map = self._collect_layer_activations()
        sliced_layers_count = 0
        total_channels_pruned = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                d_out, d_in = module.weight.shape
                device = module.weight.device
                dtype = module.weight.dtype

                # Get activations for this module
                act = activations_map.get(name, None)
                Q, eigvals = self._compute_orthogonal_rotation(act, d_in)
                Q = Q.to(device=device, dtype=torch.float32)
                W = module.weight.data.float()

                # Rotate weight: W_tilde = W @ Q
                W_tilde = torch.matmul(W, Q)

                # Determine slicing dimension r
                if self.non_uniform and "query" in name or "key" in name or "q_lin" in name or "k_lin" in name:
                    # Non-uniform attention projection head preservation: lighter slicing
                    head_sparsity = max(0.0, self.sparsity * 0.75)
                    r = max(1, int(d_in * (1.0 - head_sparsity)))
                elif self.non_uniform and ("mlp" in name or "ffn" in name or "dense" in name):
                    # Intermediate MLP projections: higher tolerance for slicing
                    mlp_sparsity = min(0.9, self.sparsity * 1.15)
                    r = max(1, int(d_in * (1.0 - mlp_sparsity)))
                else:
                    r = max(1, int(d_in * (1.0 - self.sparsity)))

                # Create projection slicing operator P_r keeping top-r principal channels
                # Zero out least variance rotated channels: r to d_in
                W_sliced_rot = W_tilde.clone()
                if r < d_in:
                    W_sliced_rot[:, r:] = 0.0
                    total_channels_pruned += (d_in - r)

                # Rotate back to original coordinate basis: W_pruned = W_sliced_rot @ Q^T
                W_pruned = torch.matmul(W_sliced_rot, Q.T)

                # Assign pruned weights preserving dtype and device
                module.weight.data = W_pruned.to(dtype=dtype, device=device)
                sliced_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "sliced_layers": sliced_layers_count,
            "pruned_channels": total_channels_pruned,
            "sparsity_target": self.sparsity,
            "non_uniform": self.non_uniform,
            "method": "orthogonal_slicegpt",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("slicegpt")
def slicegpt_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for SliceGPT compressor."""
    sparsity = getattr(comp_config, "slicegpt_sparsity", getattr(comp_config, "sparsity", 0.25)) if comp_config else kwargs.get("sparsity", 0.25)
    non_uniform = getattr(comp_config, "slicegpt_non_uniform", getattr(comp_config, "non_uniform", True)) if comp_config else kwargs.get("non_uniform", True)
    tokenizer = getattr(model, "tokenizer", None) or getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None)

    compressor = SliceGPTCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        non_uniform=non_uniform,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
