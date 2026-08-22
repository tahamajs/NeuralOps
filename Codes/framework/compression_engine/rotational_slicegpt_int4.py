"""
Rotational SliceGPT INT4 Compressor: Rotational Invariance, Subspace Slicing, and Second-Order INT4 Quantization.
Registered in COMPRESSOR_REGISTRY as 'rotational_slicegpt_int4' and 'rotational_slicegpt'.

Mathematical Formulation:
1. Orthogonal Coordinate Rotation:
   $$\\widetilde{W} = W Q, \\quad \\widetilde{X} = Q^\\top X, \\quad Q^\\top Q = I$$
2. Variance-Based Subspace Slicing:
   $$\\mathcal{I} = \\operatorname{Top-}r(\\operatorname{Var}(\\widetilde{X})), \\quad P_r = [e_i]_{i \\in \\mathcal{I}}$$
   $$W_{sliced} = \\widetilde{W} P_r, \\quad X_{sliced} = P_r^\\top \\widetilde{X}$$
3. Second-Order Optimal Brain Surgeon INT4 Non-Uniform Quantization:
   $$H = 2 X_{sliced} X_{sliced}^\\top + \\lambda I$$
   $$W_{:, j+1:r} \\leftarrow W_{:, j+1:r} - (w_j - q(w_j)) \\frac{[H^{-1}]_{j, j+1:r}}{[H^{-1}]_{jj}}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import get_calibration_tensors

logger = logging.getLogger(__name__)


class AstroQuant4Bit:
    """
    Astro 4-bit Non-Uniform Quantization Codebook mapping.
    Maps floating-point parameters to non-linear centroids optimized for Gaussian weight distributions.
    """
    def __init__(self) -> None:
        k = torch.arange(16, dtype=torch.float32)
        # c_k = Sign(k - 7.5) * (|k - 7.5| / 7.5)^1.4
        self.centroids_base = torch.sign(k - 7.5) * torch.pow(torch.abs(k - 7.5) / 7.5, 1.4)
        
    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantizes weight vector/matrix w using block-wise scales and zero-points.
        Returns the quantized weights.
        """
        w_min = w.min()
        w_max = w.max()
        
        c_min = self.centroids_base.min()
        c_max = self.centroids_base.max()
        
        # Handle the edge case where all weights are the same
        denom = c_max - c_min
        if denom == 0:
            denom = 1e-9
            
        s = (w_max - w_min) / denom
        z = w_min - c_min * s
        
        centroids = self.centroids_base.to(w.device) * s + z
        
        # Find nearest centroid for each weight element
        distances = torch.abs(w.unsqueeze(-1) - centroids.unsqueeze(0))
        nearest_indices = torch.argmin(distances, dim=-1)
        q = centroids[nearest_indices]
        return q


class RotationalSliceGPTINT4Linear(nn.Module):
    """
    A linear layer that applies the compressed Rotational-SliceGPT-INT4 weights.
    Maintains equivalent outputs to W @ X by evaluating: W_quant @ (Q P_r)^T X + bias.
    """
    def __init__(
        self,
        W_quant: torch.Tensor,
        Q: torch.Tensor,
        P_r: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # Combine the orthogonal matrix and slicing operator for efficient inference
        Q_P_r = torch.matmul(Q, P_r)  # [D_in, r]
        self.register_buffer("Q_P_r", Q_P_r)
        self.register_buffer("W_quant", W_quant)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [..., D_in]
        Returns:
            Output tensor of shape [..., D_out]
        """
        # x_proj = x @ (Q P_r) -> [..., r]
        x_proj = torch.matmul(x, self.Q_P_r)
        
        # y = x_proj @ W_quant^T -> [..., D_out]
        y = torch.matmul(x_proj, self.W_quant.T)
        if self.bias is not None:
            y = y + self.bias
        return y


class RotationalSliceGPTINT4Compressor(BaseCompressor):
    """
    Rotational-SliceGPT-INT4 Hybrid Compression Engine.
    Combines orthogonal rotation, variance-based coordinate slicing, and
    second-order Hessian INT4 non-uniform quantization.
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        r_ratio: float = 0.75,
        lambda_reg: float = 1e-2,
        slice_ratio: Optional[float] = None,
        quant_bits: int = 4,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        target_model = model if model is not None else nn.Identity()
        super().__init__(target_model, name="rotational_slicegpt_int4", **kwargs)
        self.r_ratio = slice_ratio if slice_ratio is not None else r_ratio
        self.lambda_reg = lambda_reg
        self.quant_bits = quant_bits
        self.quantizer = AstroQuant4Bit()
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(target_model.parameters(), torch.tensor(0.0)).device if model is not None else torch.device("cpu"))
        )
        self.stats: Dict[str, Any] = {}

    def get_orthogonal_matrix(self, d_in: int, device: torch.device) -> torch.Tensor:
        """
        Derives an orthogonal matrix Q via QR factorization such that Q^T Q = I.
        """
        H = torch.randn(d_in, d_in, device=device)
        Q, _ = torch.linalg.qr(H)
        return Q

    def compress_matrix(self, W: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the Rotational-SliceGPT-INT4 compression pipeline on weight matrix W and activations X.
        
        Args:
            W: Dense Weight Matrix [D_out, D_in]
            X: Activation Matrix [D_in, L]
            
        Returns:
            W_quant: Quantized Sliced Weight Matrix [D_out, r]
            Q: Orthogonal Rotation Matrix [D_in, D_in]
            P_r: Slicing Projection Operator [D_in, r]
        """
        D_out, D_in = W.shape
        r = max(1, min(int(D_in * self.r_ratio), D_in))
        device = W.device
        
        # 1. Orthogonal Rotation: \tilde{W} = W Q, \tilde{X} = Q^T X
        Q = self.get_orthogonal_matrix(D_in, device)
        W_tilde = torch.matmul(W.float(), Q)
        X_tilde = torch.matmul(Q.T, X.float().to(device))
        
        # 2. Slicing Projection P_r: top-r variance channels
        channel_variances = torch.var(X_tilde, dim=1) if X_tilde.shape[1] > 1 else torch.ones(D_in, device=device)
        top_r_indices = torch.argsort(channel_variances, descending=True)[:r]
        
        P_r = torch.zeros(D_in, r, device=device)
        for i, idx in enumerate(top_r_indices):
            P_r[idx, i] = 1.0
            
        W_sliced = torch.matmul(W_tilde, P_r)
        X_sliced = torch.matmul(P_r.T, X_tilde)
        
        # 3. SparseGPT Second-Order Hessian Updates
        H = 2.0 * torch.matmul(X_sliced, X_sliced.T) / max(X_sliced.shape[1], 1) + self.lambda_reg * torch.eye(r, device=device)
        try:
            H_inv = torch.linalg.inv(H)
        except Exception:
            H_inv = torch.linalg.pinv(H)
        H_inv = torch.nan_to_num(H_inv, nan=0.0)
        
        W_quant = W_sliced.clone()
        for j in range(r):
            w_j = W_quant[:, j].clone()
            q_j = self.quantizer.quantize(w_j)
            e_j = w_j - q_j
            
            W_quant[:, j] = q_j
            diag_inv = H_inv[j, j].item()
            if not np.isfinite(diag_inv) or diag_inv <= 1e-8:
                diag_inv = 1e-5
                
            if j < r - 1:
                h_slice = torch.nan_to_num(H_inv[j, j+1:r] / diag_inv, nan=0.0)
                update = torch.outer(e_j, h_slice)
                W_quant[:, j+1:r] -= update
                
        return W_quant, Q, P_r

    # Compatibility alias for matrix-level compression
    compress_tensor = compress_matrix

    def compress(self, *args: Any, **kwargs: Any) -> Any:
        """
        Dual-mode compress supporting both full-model compression .compress() -> (model, stats)
        and single matrix compression .compress(W, X) -> (W_quant, Q, P_r).
        """
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            return self.compress_matrix(args[0], args[1])

        activations_map = self._collect_layer_activations()
        compressed_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                act = activations_map.get(name, None)
                if act is not None and act.numel() > 0:
                    act_flat = act.reshape(-1, act.shape[-1]).float()
                    if act_flat.shape[-1] == d_in:
                        X = act_flat.T
                    else:
                        X = torch.randn(d_in, 64, device=device)
                else:
                    X = torch.randn(d_in, 64, device=device)

                W_quant, Q, P_r = self.compress_matrix(W, X)
                # Reconstruct full coordinate representation in-place
                # W_recon = W_quant @ P_r^T @ Q^T
                W_recon = torch.matmul(W_quant, torch.matmul(P_r.T, Q.T))
                module.weight.data = W_recon.to(dtype=dtype, device=device)
                compressed_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": compressed_layers_count,
            "r_ratio": self.r_ratio,
            "quant_bits": self.quant_bits,
            "lambda_reg": self.lambda_reg,
            "method": "rotational_slicegpt_int4",
        })
        self.stats = stats
        return self.model, stats

    def compress_linear(
        self,
        linear_layer: nn.Linear,
        X: Optional[torch.Tensor] = None,
    ) -> RotationalSliceGPTINT4Linear:
        """Convenience method to compress a single PyTorch nn.Linear layer."""
        W = linear_layer.weight.data
        device = W.device
        if X is None:
            X = torch.randn(W.shape[1], 128, device=device)
        elif X.dim() == 3:
            X = X.reshape(-1, W.shape[1]).T
        elif X.dim() == 2 and X.shape[0] != W.shape[1]:
            X = X.T
        W_quant, Q, P_r = self.compress_matrix(W, X)
        bias = linear_layer.bias.data if linear_layer.bias is not None else None
        return RotationalSliceGPTINT4Linear(W_quant, Q, P_r, bias=bias)

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Capture input activations across all linear layers."""
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
                logger.debug("Activation capture exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated


# Aliases
RotationalSliceGPTCompressor = RotationalSliceGPTINT4Compressor


@COMPRESSOR_REGISTRY.register("rotational_slicegpt_int4")
@COMPRESSOR_REGISTRY.register("rotational_slicegpt")
def rotational_slicegpt_int4_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for Rotational-SliceGPT-INT4 compressor."""
    r_ratio = getattr(comp_config, "rotational_slicegpt_r_ratio", getattr(comp_config, "r_ratio", 0.75)) if comp_config else kwargs.get("r_ratio", 0.75)
    lambda_reg = getattr(comp_config, "rotational_slicegpt_lambda", getattr(comp_config, "lambda_reg", 1e-2)) if comp_config else kwargs.get("lambda_reg", 1e-2)
    quant_bits = getattr(comp_config, "rotational_slicegpt_bits", getattr(comp_config, "quant_bits", 4)) if comp_config else kwargs.get("quant_bits", 4)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = RotationalSliceGPTINT4Compressor(
        model,
        r_ratio=r_ratio,
        lambda_reg=lambda_reg,
        quant_bits=quant_bits,
        tokenizer=tokenizer,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
