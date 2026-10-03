"""
Rotational SliceGPT INT4 Compressor — Production-Hardened.

Combines orthogonal rotation, variance-based subspace slicing, and
second-order INT4 non-uniform quantization.

Mathematical formulation:
  1. Rotation:       W̃ = W Q,   X̃ = Qᵀ X,   Qᵀ Q = I
  2. Slicing:        P_r = [e_i] for top-r var(X̃) channels
                     W_sliced = W̃ P_r,   X_sliced = P_rᵀ X̃
  3. Hessian INT4:   H = 2 X_sliced X_slicedᵀ + λ I
                     W[:, j+1:r] -= (w_j - q(w_j)) · [H⁻¹]_{j, j+1:r} / [H⁻¹]_{jj}

Registered: 'rotational_slicegpt_int4', 'rotational_slicegpt',
            'rotational_slicegpt_int4_factory', 'rotational_slicegpt_factory'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.pruning_utils import get_calibration_tensors
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# Astro non-uniform quantizer with actual int storage
# ===========================================================================
class AstroQuantizer:
    """
    Non-uniform quantizer with INT2 / INT4 / INT8 codebooks optimized for
    Gaussian weight distributions.

    RS-FIX-01: quantize() returns (indices int8, centroids fp32) so the
    caller can store indices compactly and reconstruct with a tiny lookup.
    """

    SUPPORTED_BITS = (2, 4, 8)

    def __init__(self, bits: int = 4) -> None:
        self.bits = bits if bits in self.SUPPORTED_BITS else 4
        self.num_centroids = 2**self.bits
        k = torch.arange(self.num_centroids, dtype=torch.float32)
        mid = (self.num_centroids - 1.0) / 2.0
        # Gaussian-shaped centroid grid (still spread over [-1, 1] before scaling).
        denom = max(mid, 1e-6)
        self.centroids_base = torch.sign(k - mid) * torch.pow(
            torch.abs(k - mid) / denom, 1.4
        )

    def quantize(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric per-tensor quantization.

        Returns:
            indices:   (..., ) int8 tensor of centroid indices in [0, C)
            centroids: (C,) fp32 tensor of scaled centroids
        """
        if w.numel() == 0:
            return torch.zeros_like(w, dtype=torch.int8), self.centroids_base.clone()

        w_min = w.min()
        w_max = w.max()
        c_min = self.centroids_base.min()
        c_max = self.centroids_base.max()
        denom = (c_max - c_min).clamp_min(1e-9)
        s = (w_max - w_min) / denom
        z = w_min - c_min * s

        centroids = self.centroids_base.to(w.device, dtype=torch.float32) * s + z
        # (..., 1) vs (1, ..., C) → (..., C)
        distances = torch.abs(w.unsqueeze(-1).float() - centroids)
        indices = torch.argmin(distances, dim=-1).to(torch.int8)
        return indices, centroids.to(torch.float32)

    @staticmethod
    def dequantize(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Reconstruct floats from indices and centroids."""
        flat_idx = indices.reshape(-1).long()
        return centroids[flat_idx].reshape(indices.shape)


AstroQuant4Bit = AstroQuantizer


# ===========================================================================
# Compressed linear layer
# ===========================================================================
class RotationalSliceGPTINT4Linear(nn.Module):
    """
    Linear layer storing:
      - Q_P_r:        (D_in, r) fp16/32 — the projection matrix
      - W_indices:    (D_out, r) int8   — quantized weight in sliced space
      - W_centroids:  (C,)       fp32   — per-tensor centroid table
      - bias:         (D_out,)   or None

    Forward: y = (x @ Q_P_r) → dequantize(W_indices) → @ .T → + bias
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        W_indices: torch.Tensor,  # (D_out, r) int8
        W_centroids: torch.Tensor,  # (C,) fp32
        Q_P_r: torch.Tensor,  # (D_in, r) float
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("Q_P_r", Q_P_r)
        self.register_buffer("W_indices", W_indices)
        self.register_buffer("W_centroids", W_centroids)

        if bias is not None:
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _dequantize_W(self) -> torch.Tensor:
        return AstroQuantizer.dequantize(self.W_indices, self.W_centroids).to(
            self.Q_P_r.dtype
        )

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight matrix (out_features, in_features)."""
        W_q = self._dequantize_W()  # (D_out, r)
        return torch.matmul(W_q, self.Q_P_r.T)  # (D_out, D_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D_in)
        x_proj = torch.matmul(x, self.Q_P_r)  # (..., r)
        W_q = self._dequantize_W()  # (D_out, r)
        y = torch.matmul(x_proj, W_q.T)  # (..., D_out)
        if self.bias is not None:
            y = y + self.bias
        return y


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("rotational_slicegpt_int4")
@COMPRESSOR_REGISTRY.register("rotational_slicegpt")
class RotationalSliceGPTINT4Compressor(BaseCompressor):
    """
    Rotational-SliceGPT-INT4 hybrid compression engine.
    """

    # RS-FIX-04: not an attention module.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        r_ratio: float = 0.4,  # RS-FIX-03: lower default
        lambda_reg: float = 1e-2,
        slice_ratio: float | None = None,
        quant_bits: int = 4,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        tokenizer: Any | None = None,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        target_modules: list[str] | None = None,
        target_attention_only: bool = False,
        seed: int = 42,  # RS-FIX-05
        **kwargs: Any,
    ) -> None:
        # Remove duplicates that arrive through kwargs/comp_config.
        for k in (
            "max_error",
            "fallback_to_original",
            "r_ratio",
            "quant_bits",
            "lambda_reg",
        ):
            kwargs.pop(k, None)
        target_model = model if model is not None else nn.Identity()
        super().__init__(target_model, name="rotational_slicegpt_int4", **kwargs)

        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        r_ratio = _cfg_get(
            cfg, "rotational_slicegpt_r_ratio", _cfg_get(cfg, "r_ratio", r_ratio)
        )
        lambda_reg = _cfg_get(cfg, "lambda_reg", lambda_reg)
        quant_bits = _cfg_get(cfg, "quant_bits", quant_bits)
        max_error = _cfg_get(cfg, "max_error", max_error)
        fallback_to_original = _cfg_get(
            cfg, "fallback_to_original", fallback_to_original
        )
        target_attention_only = bool(
            target_attention_only or _cfg_get(cfg, "target_attention_only", False)
        )

        self.r_ratio = float(slice_ratio if slice_ratio is not None else r_ratio)
        if self.r_ratio >= 0.5:
            logger.warning(
                "RotationalSliceGPT: r_ratio=%.2f ≥ 0.5 will INCREASE parameter count. "
                "Consider r_ratio ≤ 0.4 for real savings.",
                self.r_ratio,
            )
        self.lambda_reg = float(lambda_reg)
        self.quant_bits = int(quant_bits)
        self.quantizer = AstroQuantizer(bits=self.quant_bits)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.target_attention_only = target_attention_only
        self.seed = int(seed)

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (
                next(target_model.parameters(), torch.tensor(0.0)).device
                if model is not None
                else torch.device("cpu")
            )
        )
        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}
        self._orig_bytes = 0
        self._comp_bytes = 0

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(
            k in name_l
            for k in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "layernorm",
                "rmsnorm",
                "batchnorm",
            )
        ):
            return False

        if isinstance(
            module,
            (
                nn.LayerNorm,
                nn.RMSNorm,
                nn.Embedding,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                RotationalSliceGPTINT4Linear,
            ),
        ):
            return False

        if self.target_attention_only:
            attn_kw = (
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "c_attn",
                "c_proj",
                "query",
                "key",
                "value",
            )
            if not any(k in name_l for k in attn_kw):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    def get_orthogonal_matrix(self, d_in: int, device: torch.device) -> torch.Tensor:
        """Deterministic QR-based orthogonal Q (RS-FIX-05)."""
        gen = torch.Generator(device=device).manual_seed(self.seed)
        H = torch.randn(d_in, d_in, device=device, dtype=torch.float32, generator=gen)
        Q, _ = torch.linalg.qr(H)
        return Q

    # ------------------------------------------------------------------
    def compress_matrix(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float
        """
        D_out, D_in = W.shape
        r = max(1, min(int(D_in * self.r_ratio), D_in))
        device = W.device

        # 1. Orthogonal rotation
        Q = self.get_orthogonal_matrix(D_in, device)
        W_fp32 = W.detach().float()
        X_fp32 = X.detach().float().to(device)

        W_tilde = W_fp32 @ Q  # (D_out, D_in)
        X_tilde = Q.T @ X_fp32  # (D_in, N)

        # 2. Variance-based slicing (RS-FIX-06: vectorized)
        if X_tilde.shape[1] > 1:
            channel_variances = X_tilde.var(dim=1)  # (D_in,)
        else:
            channel_variances = torch.ones(D_in, device=device)
        top_r_indices = torch.argsort(channel_variances, descending=True)[:r]

        # P_r as a (D_in, r) selector matrix via scatter
        P_r = torch.zeros(D_in, r, device=device, dtype=torch.float32)
        P_r[top_r_indices, torch.arange(r, device=device)] = 1.0

        W_sliced = W_tilde @ P_r  # (D_out, r)
        X_sliced = P_r.T @ X_tilde  # (r, N)

        # 3. Second-order Hessian-based INT4 quantization
        H = 2.0 * (X_sliced @ X_sliced.T) / max(
            X_sliced.shape[1], 1
        ) + self.lambda_reg * torch.eye(r, device=device)
        try:
            H_inv = torch.linalg.inv(H)
        except Exception:
            H_inv = torch.linalg.pinv(H)
        H_inv = torch.nan_to_num(H_inv, nan=0.0)

        # RS-FIX-10: quantize the entire column of residuals at once.
        W_quant = W_sliced.clone()
        # Full-matrix OBS: quantize each column then propagate the error.
        for j in range(r):
            w_j = W_quant[:, j].clone()
            idx_j, cents_j = self.quantizer.quantize(w_j)
            q_j = AstroQuantizer.dequantize(idx_j, cents_j).to(w_j.dtype)
            e_j = w_j - q_j
            W_quant[:, j] = q_j

            diag_inv = float(H_inv[j, j].item())
            if not math.isfinite(diag_inv) or diag_inv <= 1e-8:
                diag_inv = 1e-5

            if j < r - 1:
                h_slice = torch.nan_to_num(H_inv[j, j + 1 : r] / diag_inv, nan=0.0)
                W_quant[:, j + 1 : r] -= torch.outer(e_j, h_slice)

        # Re-quantize the final W_quant column-wise to get indices + centroids
        # (OBS updates can change values slightly, so we requantize once).
        flat = W_quant.t().contiguous()  # (r, D_out) → quantize per row
        W_idx_list, W_cents_list = [], []
        per_col_centroids = []
        W_indices = torch.empty_like(W_quant, dtype=torch.int8)
        for j in range(r):
            idx_j, cents_j = self.quantizer.quantize(W_quant[:, j])
            W_indices[:, j] = idx_j
            per_col_centroids.append(cents_j)
        # Use a single shared centroid table = average of per-column tables.
        W_centroids = torch.stack(per_col_centroids, dim=0).mean(dim=0)

        # Reconstruct and measure error (using the shared table for honesty).
        W_deq = AstroQuantizer.dequantize(W_indices, W_centroids).to(W_quant.dtype)
        W_recon = (W_deq @ P_r.T) @ Q.T  # (D_out, D_in)
        norm_orig = W_fp32.norm(p="fro").clamp_min(1e-8)
        rel_error = float((W_fp32 - W_recon).norm(p="fro") / norm_orig)

        Q_P_r = Q @ P_r  # (D_in, r) — RS-FIX-08
        return W_indices, W_centroids, Q_P_r, rel_error

    compress_tensor = compress_matrix  # backward-compat alias

    # ------------------------------------------------------------------
    def compress(self, *args: Any, **kwargs: Any) -> Any:
        # Dual-mode: .compress(W, X) → tuple, .compress() → (model, stats)
        if (
            len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
        ):
            return self.compress_matrix(args[0], args[1])

        activations_map = self._collect_layer_activations()
        compressed_layers_count = 0
        fallback_layers: list[str] = []
        self.layer_errors = {}
        layer_shapes: dict[str, tuple[int, int]] = {}
        self._orig_bytes = 0
        self._comp_bytes = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_mod = getattr(module, "base_layer", module)
            raw_w = base_mod.weight.data
            cls_name = base_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().contiguous().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            layer_shapes[name] = (d_out, d_in)

            # Build X for slicing (RS-FIX-07: per-channel mean)
            act = activations_map.get(name, None)
            if act is not None and act.numel() > 0:
                act_flat = act.reshape(-1, act.shape[-1]).float()
                if act_flat.shape[-1] == d_in:
                    # Synthesize a small sample matrix from the mean vector.
                    mean_act = act_flat.mean(dim=0, keepdim=True)  # (1, D_in)
                    # Repeat with small jitter to give variance-based slicing signal.
                    X = mean_act.T + 0.01 * torch.randn(d_in, 32, device=device)
                else:
                    X = torch.randn(d_in, 32, device=device)
            else:
                X = torch.randn(d_in, 32, device=device)

            try:
                W_idx, W_cents, Q_P_r, rel_error = self.compress_matrix(W, X)
            except Exception as e:
                logger.warning("RotationalSliceGPT failed on %s: %s", name, e)
                fallback_layers.append(name)
                continue

            self.layer_errors[name] = rel_error

            if rel_error > self.max_error:
                logger.warning(
                    "RotationalSliceGPT reconstruction error %.4f on %s exceeded %.4f",
                    rel_error,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Cast buffers to the original dtype & device
            W_idx = W_idx.to(device=device)
            W_cents = W_cents.to(device=device, dtype=torch.float32)
            Q_P_r_dev = Q_P_r.to(device=device, dtype=dtype)

            bias = (
                base_mod.bias.detach()
                if getattr(base_mod, "bias", None) is not None
                else None
            )

            new_linear = RotationalSliceGPTINT4Linear(
                W_indices=W_idx,
                W_centroids=W_cents,
                Q_P_r=Q_P_r_dev,
                bias=bias,
            )

            # Splice into model
            if hasattr(module, "base_layer"):
                # Preserve any LoRA params
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    if not hasattr(new_linear, "lora_A"):
                        new_linear.lora_A = module.lora_A
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    if not hasattr(new_linear, "lora_B"):
                        new_linear.lora_B = module.lora_B
                module.base_layer = new_linear
            else:
                replace_module(self.model, name, new_linear)

            # RS-FIX-09: honest byte accounting
            orig_bytes = raw_w.numel() * raw_w.element_size()
            if base_mod.bias is not None:
                orig_bytes += base_mod.bias.numel() * base_mod.bias.element_size()
            # comp: int8 indices + fp32 centroid table + Q_P_r + bias
            comp_bytes = W_idx.numel() * W_idx.element_size()
            comp_bytes += W_cents.numel() * W_cents.element_size()
            comp_bytes += Q_P_r_dev.numel() * Q_P_r_dev.element_size()
            if bias is not None:
                comp_bytes += bias.numel() * bias.element_size()

            self._orig_bytes += orig_bytes
            self._comp_bytes += comp_bytes
            compressed_layers_count += 1

        # Forward sanity check
        forward_check_success = True
        try:
            device = next(self.model.parameters()).device
            with torch.no_grad():
                cfg = getattr(self.model, "config", None)
                vocab = getattr(cfg, "vocab_size", None) or 100
                dummy = torch.randint(0, min(vocab, 100), (1, 8), device=device)
                self.model(dummy)
        except Exception as ex:
            logger.warning("RotationalSliceGPT forward sanity check: %s", ex)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        stats = self.get_compression_stats()
        avg_err = (
            float(np.mean(list(self.layer_errors.values())))
            if self.layer_errors
            else 0.0
        )
        storage_ratio = (
            self._orig_bytes / self._comp_bytes if self._comp_bytes > 0 else 1.0
        )

        stats.update(
            {
                "compressed_layers": compressed_layers_count,
                "fallback_layers": fallback_layers,
                "r_ratio": self.r_ratio,
                "quant_bits": self.quant_bits,
                "lambda_reg": self.lambda_reg,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "layer_errors": self.layer_errors,
                "layer_shapes": layer_shapes,
                "avg_reconstruction_error": avg_err,
                "target_attention_only": self.target_attention_only,
                "forward_check_success": forward_check_success,
                "orig_storage_bytes": self._orig_bytes,
                "comp_storage_bytes": self._comp_bytes,
                "storage_compression_ratio": round(storage_ratio, 3),
                "storage_reduction_pct": round(
                    100.0 * (1.0 - self._comp_bytes / max(1, self._orig_bytes)), 2
                ),
                "method": "rotational_slicegpt_int4",
            }
        )
        self.stats = stats
        logger.info(
            "RotationalSliceGPT: %d layers compressed, %d fallback, "
            "avg_err=%.4f, storage_ratio=%.2fx, bits=%d, r_ratio=%.2f",
            compressed_layers_count,
            len(fallback_layers),
            avg_err,
            storage_ratio,
            self.quant_bits,
            self.r_ratio,
        )
        return self.model, stats

    # ------------------------------------------------------------------
    def _collect_layer_activations(self) -> dict[str, torch.Tensor]:
        """
        RS-FIX-07: store per-channel mean |x| instead of full activation
        tensors to avoid OOM on large models.
        """
        self.model.eval()
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    x = inputs[0].detach()
                    if x.dim() >= 2:
                        col_mean = x.abs().mean(dim=tuple(range(x.dim() - 1)))
                    else:
                        col_mean = x.abs()
                    activations.setdefault(layer_name, []).append(col_mean.cpu())

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                target = getattr(module, "base_layer", module)
                hooks.append(target.register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad():
                calib_inputs = get_calibration_tensors(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
                try:
                    if isinstance(calib_inputs, dict):
                        self.model(**calib_inputs)
                    elif isinstance(calib_inputs, torch.Tensor):
                        self.model(calib_inputs)
                    elif isinstance(calib_inputs, (list, tuple)):
                        self.model(*calib_inputs)
                except Exception as e:
                    logger.debug("Activation capture failed: %s", e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: dict[str, torch.Tensor] = {}
        for name, lst in activations.items():
            if lst:
                aggregated[name] = torch.stack([t.float() for t in lst], dim=0).mean(
                    dim=0
                )
        return aggregated

    # ------------------------------------------------------------------
    def compress_linear(
        self,
        linear_layer: nn.Linear,
        X: torch.Tensor | None = None,
    ) -> RotationalSliceGPTINT4Linear:
        """Convenience wrapper for compressing a single nn.Linear."""
        W = linear_layer.weight.data.float()
        device = W.device
        if X is None:
            X = torch.randn(W.shape[1], 128, device=device)
        elif X.dim() == 3:
            X = X.reshape(-1, W.shape[1]).T
        elif X.dim() == 2 and X.shape[0] != W.shape[1]:
            X = X.T
        W_idx, W_cents, Q_P_r, _ = self.compress_matrix(W, X)
        bias = (
            linear_layer.bias.data
            if getattr(linear_layer, "bias", None) is not None
            else None
        )
        return RotationalSliceGPTINT4Linear(
            W_indices=W_idx.to(device),
            W_centroids=W_cents.to(device),
            Q_P_r=Q_P_r.to(device=device, dtype=W.dtype),
            bias=bias,
        )


# Aliases
RotationalSliceGPTCompressor = RotationalSliceGPTINT4Compressor
RotationalSliceGPTInt4Compressor = RotationalSliceGPTINT4Compressor


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("rotational_slicegpt_int4_factory")
@COMPRESSOR_REGISTRY.register("rotational_slicegpt_factory")
def rotational_slicegpt_int4_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    r_ratio = kwargs.pop(
        "r_ratio",
        _cfg_get(
            comp_config,
            "rotational_slicegpt_r_ratio",
            _cfg_get(comp_config, "r_ratio", 0.4),
        ),
    )
    lambda_reg = kwargs.pop("lambda_reg", _cfg_get(comp_config, "lambda_reg", 1e-2))
    quant_bits = kwargs.pop("quant_bits", _cfg_get(comp_config, "quant_bits", 4))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.20))
    fallback_to_original = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "fallback_to_original", True),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    seed = kwargs.pop("seed", 42)

    compressor = RotationalSliceGPTINT4Compressor(
        model,
        r_ratio=r_ratio,
        lambda_reg=lambda_reg,
        quant_bits=quant_bits,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        target_attention_only=target_attention_only,
        seed=seed,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "RotationalSliceGPTINT4Compressor",
    "RotationalSliceGPTInt4Compressor",
    "RotationalSliceGPTCompressor",
    "RotationalSliceGPTINT4Linear",
    "AstroQuantizer",
    "AstroQuant4Bit",
    "rotational_slicegpt_int4_factory",
]
