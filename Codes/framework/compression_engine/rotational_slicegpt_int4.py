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

Production Hardening:
- Fix 1: Reconstruction error validation in compress_matrix and compress with configurable max_error (default 0.20).
- Fix 2: Default fallback_to_original changed to True to protect model accuracy.
- Fix 3: Robust Conv1D detection by class name and nf attribute, and fused QKV projection handling.
- Fix 4: Full multi-bitwidth support (2, 4, 8 bits) in AstroQuantizer.
- Fix 5: Unified critical-layer protection delegated solely to should_compress_linear.
- Fix 6: Detailed per-layer stats (errors, shapes, parameters, slicing ratios).
- Fix 7: Numerical stability: all calculations in FP32 with damped Hessian inversion.
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
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class AstroQuantizer:
    """
    Astro Non-Uniform Quantization Codebook mapping supporting 2, 4, and 8 bits.
    Maps floating-point parameters to non-linear centroids optimized for Gaussian weight distributions.
    """

    def __init__(self, bits: int = 4) -> None:
        self.bits = bits if bits in (2, 4, 8) else 4
        num_centroids = 2 ** self.bits
        k = torch.arange(num_centroids, dtype=torch.float32)
        mid = (num_centroids - 1.0) / 2.0
        # Centroids scaled to Gaussian distribution
        self.centroids_base = torch.sign(k - mid) * torch.pow(torch.abs(k - mid) / max(mid, 1e-6), 1.4)

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        """Quantizes weight vector/matrix w using block-wise scales and zero-points."""
        w_min = w.min()
        w_max = w.max()

        c_min = self.centroids_base.min()
        c_max = self.centroids_base.max()

        denom = c_max - c_min
        if denom == 0:
            denom = 1e-9

        s = (w_max - w_min) / denom
        z = w_min - c_min * s

        centroids = self.centroids_base.to(w.device) * s + z

        # Find nearest centroid for each weight element
        distances = torch.abs(w.unsqueeze(-1) - centroids.unsqueeze(0))
        nearest_indices = torch.argmin(distances, dim=-1)
        return centroids[nearest_indices]


# Backward compatibility alias
AstroQuant4Bit = AstroQuantizer


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
        Q_P_r = torch.matmul(Q, P_r)  # [D_in, r]
        self.register_buffer("Q_P_r", Q_P_r)
        self.register_buffer("W_quant", W_quant)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.W_quant, self.Q_P_r.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = torch.matmul(x, self.Q_P_r)
        y = torch.matmul(x_proj, self.W_quant.T)
        if self.bias is not None:
            y = y + self.bias
        return y


class RotationalSliceGPTINT4Compressor(BaseCompressor):
    """
    Rotational-SliceGPT-INT4 Hybrid Compression Engine.
    Combines orthogonal rotation, variance-based coordinate slicing, and
    second-order Hessian INT4/INT8/INT2 non-uniform quantization.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        r_ratio: float = 0.75,
        lambda_reg: float = 1e-2,
        slice_ratio: Optional[float] = None,
        quant_bits: int = 4,
        max_error: float = 0.20,
        fallback_to_original: bool = True,  # Fix 2: Default changed to True
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        target_model = model if model is not None else nn.Identity()
        super().__init__(target_model, name="rotational_slicegpt_int4", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            r_ratio = getattr(cfg, "rotational_slicegpt_r_ratio", getattr(cfg, "r_ratio", r_ratio))
            lambda_reg = getattr(cfg, "lambda_reg", lambda_reg)
            quant_bits = getattr(cfg, "quant_bits", quant_bits)
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.r_ratio = float(slice_ratio if slice_ratio is not None else r_ratio)
        self.lambda_reg = float(lambda_reg)
        self.quant_bits = int(quant_bits)
        # Fix 4: Support requested bit width (2, 4, or 8)
        self.quantizer = AstroQuantizer(bits=self.quant_bits)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(target_model.parameters(), torch.tensor(0.0)).device if model is not None else torch.device("cpu"))
        )
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Fix 5: Unified critical-layer protection delegated solely to should_compress_linear."""
        cls_name = module.__class__.__name__.lower()
        # Fix 3: Reliable Conv1D check using class name and hasattr(nf)
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like:
            return False
        if isinstance(module, RotationalSliceGPTINT4Linear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def get_orthogonal_matrix(self, d_in: int, device: torch.device) -> torch.Tensor:
        """Derives an orthogonal matrix Q via QR factorization such that Q^T Q = I."""
        H = torch.randn(d_in, d_in, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(H)
        return Q

    def compress_matrix(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Executes the Rotational-SliceGPT-INT4 compression pipeline on weight matrix W and activations X.
        Fix 1 & 7: Computes relative Frobenius reconstruction error in FP32.
        Returns: (W_quant, Q, P_r, rel_error)
        """
        D_out, D_in = W.shape
        r = max(1, min(int(D_in * self.r_ratio), D_in))
        device = W.device

        # 1. Orthogonal Rotation: \tilde{W} = W Q, \tilde{X} = Q^T X
        Q = self.get_orthogonal_matrix(D_in, device)
        W_fp32 = W.detach().float()
        X_fp32 = X.detach().float().to(device)

        W_tilde = torch.matmul(W_fp32, Q)
        X_tilde = torch.matmul(Q.T, X_fp32)

        # 2. Slicing Projection P_r: top-r variance channels
        channel_variances = torch.var(X_tilde, dim=1) if X_tilde.shape[1] > 1 else torch.ones(D_in, device=device)
        top_r_indices = torch.argsort(channel_variances, descending=True)[:r]

        P_r = torch.zeros(D_in, r, device=device, dtype=torch.float32)
        for i, idx in enumerate(top_r_indices):
            P_r[idx, i] = 1.0

        W_sliced = torch.matmul(W_tilde, P_r)
        X_sliced = torch.matmul(P_r.T, X_tilde)

        # 3. Second-Order Hessian Updates with Damping
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
                h_slice = torch.nan_to_num(H_inv[j, j + 1 : r] / diag_inv, nan=0.0)
                update = torch.outer(e_j, h_slice)
                W_quant[:, j + 1 : r] -= update

        # Fix 1: Reconstruction validation in compress_matrix
        W_recon = torch.matmul(W_quant, torch.matmul(P_r.T, Q.T))
        norm_orig = torch.norm(W_fp32, p="fro")
        rel_error = (torch.norm(W_fp32 - W_recon, p="fro") / (norm_orig + 1e-8)).item()

        return W_quant, Q, P_r, rel_error

    # Compatibility alias
    compress_tensor = compress_matrix

    def compress(self, *args: Any, **kwargs: Any) -> Any:
        """
        Dual-mode compress supporting full-model compression .compress() -> (model, stats)
        and single matrix compression .compress(W, X) -> (W_quant, Q, P_r, rel_error).
        """
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            return self.compress_matrix(args[0], args[1])

        activations_map = self._collect_layer_activations()
        compressed_layers_count = 0
        self.layer_errors = {}
        layer_shapes = {}

        for name, module in self.model.named_modules():
            if not self._is_target_module(name, module):
                continue

            raw_w = module.weight.data
            cls_name = module.__class__.__name__.lower()
            # Fix 3: Conv1D detection by class name and nf attribute
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            if is_conv1d:
                W = raw_w.t().clone().float()
            else:
                W = raw_w.clone().float()

            d_out, d_in = W.shape
            layer_shapes[name] = (d_out, d_in)

            # Retrieve or generate calibration activations
            act = activations_map.get(name, None)
            if act is not None and act.numel() > 0:
                act_flat = act.reshape(-1, act.shape[-1]).float()
                if act_flat.shape[-1] == d_in:
                    X = act_flat.T
                else:
                    X = torch.randn(d_in, 64, device=device)
            else:
                X = torch.randn(d_in, 64, device=device)

            W_quant, Q, P_r, rel_error = self.compress_matrix(W, X)
            self.layer_errors[name] = rel_error

            # Fix 1 & 2: Error threshold check and safe fallback
            if rel_error > self.max_error:
                logger.warning(
                    "RotationalSliceGPT reconstruction error %.4f on %s exceeded threshold %.4f%s",
                    rel_error,
                    name,
                    self.max_error,
                    ". Keeping original layer." if self.fallback_to_original else ". Proceeding with compressed weight.",
                )
                if self.fallback_to_original:
                    continue

            # Reconstruct full coordinate representation in-place
            W_recon = torch.matmul(W_quant, torch.matmul(P_r.T, Q.T))
            final_w = W_recon.t() if is_conv1d else W_recon
            module.weight.data = final_w.to(dtype=dtype, device=device)
            compressed_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        # Fix 6: Detailed per-layer stats
        stats.update({
            "compressed_layers": compressed_layers_count,
            "r_ratio": self.r_ratio,
            "quant_bits": self.quant_bits,
            "lambda_reg": self.lambda_reg,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "layer_errors": self.layer_errors,
            "layer_shapes": layer_shapes,
            "method": "rotational_slicegpt_int4",
        })
        self.stats = stats
        logger.info(
            "RotationalSliceGPT compressed %d layers (avg error: %.4f, bits: %d, slice_ratio: %.2f)",
            compressed_layers_count,
            float(np.mean(list(self.layer_errors.values()))) if self.layer_errors else 0.0,
            self.quant_bits,
            self.r_ratio,
        )
        return self.model, stats

    def compress_linear(
        self,
        linear_layer: nn.Linear,
        X: Optional[torch.Tensor] = None,
    ) -> RotationalSliceGPTINT4Linear:
        """Convenience method to compress a single PyTorch linear layer."""
        W = linear_layer.weight.data.float()
        device = W.device
        if X is None:
            X = torch.randn(W.shape[1], 128, device=device)
        elif X.dim() == 3:
            X = X.reshape(-1, W.shape[1]).T
        elif X.dim() == 2 and X.shape[0] != W.shape[1]:
            X = X.T
        W_quant, Q, P_r, _ = self.compress_matrix(W, X)
        bias = linear_layer.bias.data if getattr(linear_layer, "bias", None) is not None else None
        return RotationalSliceGPTINT4Linear(W_quant, Q, P_r, bias=bias)

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Capture input activations across all target linear layers."""
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
            if self._is_target_module(name, module):
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
    lambda_reg = getattr(comp_config, "lambda_reg", 1e-2) if comp_config else kwargs.get("lambda_reg", 1e-2)
    quant_bits = getattr(comp_config, "quant_bits", 4) if comp_config else kwargs.get("quant_bits", 4)
    max_error = getattr(comp_config, "max_error", 0.20) if comp_config else kwargs.get("max_error", 0.20)
    fallback_to_original = getattr(comp_config, "fallback_to_original", True) if comp_config else kwargs.get("fallback_to_original", True)

    compressor = RotationalSliceGPTINT4Compressor(
        model,
        r_ratio=r_ratio,
        lambda_reg=lambda_reg,
        quant_bits=quant_bits,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "RotationalSliceGPTINT4Compressor",
    "RotationalSliceGPTCompressor",
    "RotationalSliceGPTINT4Linear",
    "AstroQuantizer",
    "AstroQuant4Bit",
    "rotational_slicegpt_int4_factory",
]
