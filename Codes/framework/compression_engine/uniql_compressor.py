"""UniQL: Joint low-rank + uniform quantization compressor.

Registered as 'uniql', 'uniql_compressor', 'uniql_factory'.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---------------------------------------------------------------------------
# Real low-rank + quantized linear layer
# ---------------------------------------------------------------------------
class UniQLinear(nn.Module):
    """
    Replaces a Linear (or Conv1D) with a stored factorization:

        W ≈ U @ diag(S) @ Vt
        where U  : (out, r) int8, scale_u  (out,)
              S  : (r,)    int8, scale_s  scalar-per-rank
              Vt : (r, in) int8, scale_v  (in,)

    Forward reconstructs W_hat on the fly and applies F.linear / F.linear with
    transposed Conv1D weight. Numerically this is exactly the same
    reconstruction as the original UniQL, but now the storage is genuinely
    reduced from out*in fp16 to (out + in) * r int8 plus small scale vectors.

    Original bias (if any) is stored as-is unless quantize_bias=True.
    """
    is_custom_attention: bool = False   # UQ-01: compressors are not attention.

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        U_q: torch.Tensor,        # (out, r) int8
        S_q: torch.Tensor,        # (r,)    int8
        Vt_q: torch.Tensor,       # (r, in) int8
        scale_u: torch.Tensor,    # (out,)
        scale_s: torch.Tensor,    # (r,)
        scale_v: torch.Tensor,    # (in,)
        bias: Optional[torch.Tensor] = None,
        is_conv1d: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.is_conv1d = is_conv1d

        # Store quantized factors as non-trainable buffers.
        self.register_buffer("U_q", U_q.to(device=device), persistent=True)
        self.register_buffer("S_q", S_q.to(device=device), persistent=True)
        self.register_buffer("Vt_q", Vt_q.to(device=device), persistent=True)
        self.register_buffer("scale_u", scale_u.to(device=device, dtype=torch.float32), persistent=True)
        self.register_buffer("scale_s", scale_s.to(device=device, dtype=torch.float32), persistent=True)
        self.register_buffer("scale_v", scale_v.to(device=device, dtype=torch.float32), persistent=True)

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype=dtype, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self._weight_dtype = dtype

    # ---- Reconstruct in fp32 for numerical stability ----
    def _reconstruct_weight(self) -> torch.Tensor:
        U = self.U_q.float() * self.scale_u.unsqueeze(1)          # (out, r)
        S = self.S_q.float() * self.scale_s                        # (r,)
        Vt = self.Vt_q.float() * self.scale_v.unsqueeze(0)         # (r, in)
        W = (U * S.unsqueeze(0)) @ Vt                              # (out, in)
        return W.to(self._weight_dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed weight (out, in). Conv1D callers should transpose."""
        return self._reconstruct_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._reconstruct_weight()
        if self.is_conv1d:
            # Conv1D stores weight as (in, out); our W is (out, in).
            out = F.linear(x, W)
        else:
            out = F.linear(x, W)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, conv1d={self.is_conv1d}"
        )


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def _uniform_quantize(t: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Uniform min-max quantization to signed int8 (symmetric) with per-row or
    per-tensor scale. Returns (q_int8, scale, min_val_or_0).

    For our reconstruction, we use symmetric per-row scale when the tensor is
    2D, and per-tensor scale when it is 1D.
    """
    tf = t.float()
    if tf.dim() == 2:
        # per-row symmetric
        max_abs = tf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        qmax = float((2 ** (bits - 1)) - 1)
        scale = (max_abs / qmax).squeeze(1)                 # (rows,)
        q = torch.clamp(torch.round(tf / scale.unsqueeze(1)), -qmax - 1, qmax).to(torch.int8)
        return q, scale, 0.0
    elif tf.dim() == 1:
        max_abs = tf.abs().max().clamp(min=1e-8)
        qmax = float((2 ** (bits - 1)) - 1)
        scale = (max_abs / qmax).reshape(())
        q = torch.clamp(torch.round(tf / scale), -qmax - 1, qmax).to(torch.int8)
        return q, scale, 0.0
    else:
        # fallback: per-tensor symmetric
        max_abs = tf.abs().max().clamp(min=1e-8)
        qmax = float((2 ** (bits - 1)) - 1)
        scale = (max_abs / qmax).reshape(())
        q = torch.clamp(torch.round(tf / scale), -qmax - 1, qmax).to(torch.int8)
        return q, scale, 0.0


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------
@COMPRESSOR_REGISTRY.register("uniql_compressor")
@COMPRESSOR_REGISTRY.register("uniql")
class UniQLCompressor(BaseCompressor):
    """
    UniQL: Joint Truncated SVD + Uniform Quantization compressor.

    Unlike the naive "copy back" variant, this compressor installs a real
    UniQLinear module that stores int8 U, S, Vt buffers. Parameter count
    and memory footprint decrease by roughly the SVD factor
    (out + in) * r / (out * in).
    """
    # UQ-01: NOT an attention module.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        bits: int = 8,
        max_error: float = 0.20,
        max_retries: int = 3,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        quantize_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="uniql", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "uniql_rank", _cfg_get(cfg, "rank", rank))
            bits = _cfg_get(cfg, "uniql_bits", _cfg_get(cfg, "bits", bits))
            max_error = _cfg_get(cfg, "max_error", _cfg_get(cfg, "max_reconstruction_error", max_error))
            max_retries = _cfg_get(cfg, "max_retries", max_retries)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            quantize_bias = _cfg_get(cfg, "uniql_quantize_bias", quantize_bias)

        if kwargs.get("rank") is not None:
            rank = kwargs["rank"]
        if kwargs.get("bits") is not None:
            bits = kwargs["bits"]
        if kwargs.get("max_error") is not None:
            max_error = float(kwargs["max_error"])
        if kwargs.get("max_retries") is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])
        if "target_attention_only" in kwargs:
            target_attention_only = bool(kwargs["target_attention_only"])
        if "quantize_bias" in kwargs:
            quantize_bias = bool(kwargs["quantize_bias"])

        self.rank = max(1, int(rank))
        self.bits = max(2, min(16, int(bits)))   # cap at 16 bits (int8 storage assumption)
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = target_attention_only
        self.quantize_bias = bool(quantize_bias)
        self.compressed_layers: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # Protect classifier, pooler, lm_head, embeddings, norms.
        name_lower = name.lower()
        if any(k in name_lower for k in (
            "classifier", "score", "lm_head", "pooler", "embedding",
            "layernorm", "rmsnorm", "batchnorm",
        )):
            return False

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d)):
            return False

        # Skip if already compressed.
        if isinstance(module, UniQLinear):
            return False

        if self.target_attention_only:
            attn_keywords = (
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "c_attn", "c_proj", "query", "key", "value",
            )
            if not any(k in name_lower for k in attn_keywords):
                return False

        base_layer = getattr(module, "base_layer", module)
        cls_name = base_layer.__class__.__name__.lower()
        # UQ-10: Conv1D detection across transformers versions.
        is_conv1d = (
            "conv1d" in cls_name
            or (hasattr(base_layer, "nf") and hasattr(base_layer, "weight"))
        )
        is_linear_like = (
            isinstance(base_layer, nn.Linear)
            or is_conv1d
            or (
                hasattr(base_layer, "weight")
                and hasattr(base_layer.weight, "ndim")
                and base_layer.weight.ndim == 2
            )
        )
        if not is_linear_like:
            return False

        # Respect framework policy (e.g. Qwen GQA asymmetric skips).
        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    def _normalize_weight(self, base_layer: nn.Module) -> Tuple[torch.Tensor, bool]:
        """
        Return (W as (out, in) fp32, is_conv1d).
        UQ-04: explicit, not heuristic.
        """
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = (
            "conv1d" in cls_name
            or (hasattr(base_layer, "nf") and raw_w.shape[0] < raw_w.shape[1])
        )
        if is_conv1d:
            W = raw_w.t().contiguous().float()
        else:
            W = raw_w.float().contiguous()
        return W, is_conv1d

    # ------------------------------------------------------------------
    def _try_compress(self, W: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Attempt SVD + quantization with adaptive bit/rank escalation.
        UQ-05: monotonic, bounded escalation.
        Returns a dict with quantized factors + reconstruction error, or None.
        """
        out_f, in_f = W.shape
        max_r = min(out_f, in_f)
        if max_r < 1:
            return None

        norm_orig = torch.linalg.norm(W).item() + 1e-12

        # ---- SVD once, in fp32 ----
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("UniQL SVD failed: %s", e)
            return None

        # Candidate ranks: start at requested rank, expand up to max_r.
        base_r = max(1, min(self.rank, max_r))
        rank_candidates = [base_r]
        if base_r < max_r:
            rank_candidates.append(min(max_r, base_r * 2))
        if max_r not in rank_candidates:
            rank_candidates.append(max_r)

        # Candidate bits: bits, then 8, then 16 (int8 storage cap).
        bit_candidates = [self.bits]
        for b in (8, 16):
            if b > bit_candidates[-1]:
                bit_candidates.append(b)

        best = None
        best_err = float("inf")

        for cur_r in rank_candidates:
            U_r = U[:, :cur_r]
            S_r = S[:cur_r]
            Vt_r = Vt[:cur_r, :]
            for cur_bits in bit_candidates:
                # UQ-03: quantize the actual stored factors.
                U_q, scale_u, _ = _uniform_quantize(U_r, cur_bits)
                S_q, scale_s, _ = _uniform_quantize(S_r, cur_bits)
                Vt_q, scale_v, _ = _uniform_quantize(Vt_r, cur_bits)

                # Reconstruct in fp32 for error measurement.
                U_hat = U_q.float() * scale_u.unsqueeze(1)
                S_hat = S_q.float() * scale_s
                Vt_hat = Vt_q.float() * scale_v.unsqueeze(0)
                W_rec = (U_hat * S_hat.unsqueeze(0)) @ Vt_hat
                rel_err = (torch.linalg.norm(W - W_rec).item()) / norm_orig

                if rel_err < best_err:
                    best_err = rel_err
                    best = {
                        "U_q": U_q, "S_q": S_q, "Vt_q": Vt_q,
                        "scale_u": scale_u, "scale_s": scale_s, "scale_v": scale_v,
                        "rank": cur_r, "bits": cur_bits, "rel_err": rel_err,
                    }
                if rel_err <= self.max_error:
                    return best

        if best is None or best_err > self.max_error:
            return None
        return best

    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers = []
        fallback_layers: List[str] = []
        total_err = 0.0
        orig_storage_bytes = 0
        comp_storage_bytes = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_layer = getattr(module, "base_layer", module)
            raw_w = base_layer.weight.data
            orig_dtype = raw_w.dtype
            orig_device = raw_w.device

            W, is_conv1d = self._normalize_weight(base_layer)
            out_f, in_f = W.shape
            if out_f < 1 or in_f < 1:
                continue

            result = self._try_compress(W)
            if result is None:
                if self.fallback_to_original:
                    logger.info("UniQL: keeping original weights for %s", name)
                fallback_layers.append(name)
                continue

            # ---- Build the new module ----
            bias_tensor = None
            if getattr(base_layer, "bias", None) is not None and base_layer.bias is not None:
                b = base_layer.bias.data
                if self.quantize_bias:
                    bq, bs, _ = _uniform_quantize(b.float(), result["bits"])
                    b = bq.float() * bs
                bias_tensor = b.to(dtype=orig_dtype, device=orig_device)

            new_layer = UniQLinear(
                in_features=in_f,
                out_features=out_f,
                rank=result["rank"],
                U_q=result["U_q"],
                S_q=result["S_q"],
                Vt_q=result["Vt_q"],
                scale_u=result["scale_u"].to(orig_device),
                scale_s=result["scale_s"].to(orig_device),
                scale_v=result["scale_v"].to(orig_device),
                bias=bias_tensor,
                is_conv1d=is_conv1d,
                dtype=orig_dtype,
                device=orig_device,
            )
            new_layer.train(mode=module.training)

            # ---- Splice into the model ----
            if hasattr(module, "base_layer"):
                # Wrapped (e.g. LoRALinear). Keep wrapper; replace its base_layer.
                module.base_layer = new_layer
                # Freeze any remaining LoRA parameters so recovery can still
                # train them if the pipeline expects that.
                for p_name, p in module.named_parameters():
                    if "lora_" in p_name.lower() and not p_name.startswith("base_layer"):
                        p.requires_grad = False
            else:
                # Plain Linear / Conv1D: replace in the model.
                parent_name, _, child_name = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, child_name, new_layer)

            # ---- Storage accounting (honest) ----
            # UQ-07: report real bytes.
            orig_bytes = raw_w.numel() * raw_w.element_size()
            comp_bytes = (
                result["U_q"].numel() + result["S_q"].numel() + result["Vt_q"].numel()
            )  # int8: 1 byte each
            comp_bytes += (
                result["scale_u"].numel()
                + result["scale_s"].numel()
                + result["scale_v"].numel()
            ) * 4  # fp32 scales
            if bias_tensor is not None:
                orig_bytes += base_layer.bias.numel() * base_layer.bias.element_size()
                comp_bytes += bias_tensor.numel() * bias_tensor.element_size()

            orig_storage_bytes += orig_bytes
            comp_storage_bytes += comp_bytes

            self.compressed_layers.append({
                "name": name,
                "original_shape": (out_f, in_f),
                "rank": result["rank"],
                "bits": result["bits"],
                "rel_error": float(result["rel_err"]),
                "orig_bytes": orig_bytes,
                "comp_bytes": comp_bytes,
            })
            total_err += result["rel_err"]

        # ---- Stats ----
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        stats = self.get_compression_stats()
        avg_err = (total_err / len(self.compressed_layers)) if self.compressed_layers else 0.0
        ratio = (orig_storage_bytes / comp_storage_bytes) if comp_storage_bytes > 0 else 1.0

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": fallback_layers,
            "num_compressed_layers": len(self.compressed_layers),
            "num_fallback": len(fallback_layers),
            "rank": self.rank,
            "bits": self.bits,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            # UQ-07: honest accounting
            "orig_storage_bytes": orig_storage_bytes,
            "comp_storage_bytes": comp_storage_bytes,
            "storage_compression_ratio": round(ratio, 3),
            "storage_reduction_pct": round(100.0 * (1.0 - comp_storage_bytes / max(1, orig_storage_bytes)), 2),
        })

        logger.info(
            "UniQL: %d layers compressed, %d fallback, avg_err=%.4f, storage_ratio=%.2fx",
            len(self.compressed_layers), len(fallback_layers), avg_err, ratio,
        )
        return self.model, stats


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
@COMPRESSOR_REGISTRY.register("uniql_factory")
def uniql_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """UniQL factory."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "uniql_rank", _cfg_get(comp_config, "rank", 64)))
    bits = kwargs.pop("bits", _cfg_get(comp_config, "uniql_bits", _cfg_get(comp_config, "bits", 8)))
    target_attn = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    compressor = UniQLCompressor(
        model,
        rank=rank,
        bits=bits,
        comp_config=comp_config,
        target_attention_only=target_attn,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = ["UniQLinear", "UniQLCompressor", "uniql_factory"]