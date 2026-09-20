"""UniQL: Joint low‑rank + quantisation using SVD and uniform quant. Registered as 'uniql'.
Production-hardened with:
- Relative Frobenius reconstruction error validation (err <= max_error)
- Adaptive bit and rank retries
- Conv1D support (GPT-2)
- Critical layer protection (classifier, score, lm_head, pooler, embedding)
- Safe fallback preserving original weights
- Detailed per-layer error stats logging
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("uniql_compressor")
class UniQLCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    UniQL Compressor: Joint Truncated SVD and Uniform Min-Max Quantization.

    Args:
        model: Model to compress.
        rank: Target rank for low-rank SVD truncation (default 64).
        bits: Number of quantization bits (default 8).
        max_error: Maximum allowed relative Frobenius reconstruction error (default 0.20).
        max_retries: Number of retry attempts adjusting bits/rank if error exceeds max_error (default 3).
        fallback_to_original: If True, keep original module weights if error remains above max_error (default True).
        target_attention_only: If True, only compress attention projections.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        bits: int = 8,
        max_error: float = 0.20,
        max_retries: int = 3,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
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

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]
        if "bits" in kwargs and kwargs["bits"] is not None:
            bits = kwargs["bits"]
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "max_retries" in kwargs and kwargs["max_retries"] is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])
        if "target_attention_only" in kwargs:
            target_attention_only = bool(kwargs["target_attention_only"])

        self.rank = max(1, int(rank))
        self.bits = max(1, min(32, int(bits)))
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = target_attention_only
        self.quantize_bias = bool(kwargs.get("quantize_bias", False))
        self.compressed_layers: List[Dict[str, Any]] = []

    @staticmethod
    def _quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Uniform min-max quantization for FP16/FP32 stability."""
        if bits >= 32:
            return tensor.float()
        t = tensor.float()
        min_val, max_val = t.min(), t.max()
        if max_val == min_val:
            return t
        qmax = float(max((2**bits) - 1, 1))
        scale = torch.clamp((max_val - min_val) / qmax, min=1e-8)
        quant = torch.clamp(torch.round((t - min_val) / scale), 0.0, qmax) * scale + min_val
        return quant

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # Protect critical classification heads, poolers, lm_heads, and embeddings
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        if self.target_attention_only:
            attn_keywords = ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"]
            if not any(k in name.lower() for k in attn_keywords):
                return False

        base_layer = getattr(module, "base_layer", module)
        cls_name = base_layer.__class__.__name__
        is_linear_like = (
            isinstance(base_layer, nn.Linear)
            or "conv1d" in cls_name.lower()
            or hasattr(base_layer, "nf")
            or (
                hasattr(base_layer, "weight")
                and hasattr(base_layer.weight, "ndim")
                and base_layer.weight.ndim == 2
                and not isinstance(base_layer, (nn.Embedding, nn.LayerNorm, nn.RMSNorm))
            )
        )
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        return True

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the model linear/Conv1D layers using UniQL low-rank quantization."""
        self.compressed_layers = []
        fallback_layers = []
        total_error = 0.0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_layer = getattr(module, "base_layer", module)
            raw_w = base_layer.weight.data
            orig_dtype = raw_w.dtype
            orig_device = raw_w.device
            cls_name = base_layer.__class__.__name__

            is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_layer, "nf") or (
                raw_w.ndim == 2
                and hasattr(base_layer, "in_features")
                and base_layer.in_features == raw_w.shape[0]
                and raw_w.shape[0] < raw_w.shape[1]
            )

            if is_conv1d:
                W = raw_w.t().contiguous().float().contiguous()
                in_f = W.shape[1]
                out_f = W.shape[0]
            else:
                W = raw_w.float().contiguous()
                out_f, in_f = W.shape[0], W.shape[1]

            norm_orig = torch.linalg.norm(W).item() + 1e-8
            max_r = min(out_f, in_f)
            if max_r < 1:
                continue

            # Compute SVD in FP32
            try:
                if torch.cuda.is_available() and W.is_cuda:
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                else:
                    W_dev = W.to(orig_device) if torch.cuda.is_available() else W
                    U, S, Vt = torch.linalg.svd(W_dev, full_matrices=False)
            except Exception:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    U, S, Vt = torch.linalg.svd(W.cpu().float(), full_matrices=False)
                    U, S, Vt = U.to(orig_device), S.to(orig_device), Vt.to(orig_device)
                except Exception as e2:
                    logger.warning("UniQL SVD failed for %s: %s. Keeping original module.", name, e2)
                    fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cur_r = max(1, min(self.rank, max_r))
            cur_bits = self.bits
            best_w_rec = None
            best_err = float("inf")
            best_rank = cur_r
            best_bits = cur_bits

            for attempt in range(self.max_retries + 1):
                U_r = U[:, :cur_r]
                S_r = S[:cur_r]
                Vt_r = Vt[:cur_r, :]

                U_q = self._quantize(U_r, cur_bits)
                S_q = self._quantize(S_r, cur_bits)
                Vt_q = self._quantize(Vt_r, cur_bits)
                W_rec = (U_q * S_q.unsqueeze(0)) @ Vt_q

                rel_err = (torch.linalg.norm(W - W_rec).item()) / norm_orig

                if rel_err < best_err:
                    best_err = rel_err
                    best_w_rec = W_rec
                    best_rank = cur_r
                    best_bits = cur_bits

                if rel_err <= self.max_error:
                    break

                # Adaptive retry: first step up bits (e.g. 4 -> 8 -> 16 -> 32), then rank
                if cur_bits < 16:
                    cur_bits = min(16, cur_bits * 2)
                elif cur_bits < 32:
                    cur_bits = 32
                elif cur_r < max_r:
                    step = max(4, (max_r - cur_r) // max(1, self.max_retries - attempt))
                    cur_r = min(cur_r + step, max_r)

            if best_err > self.max_error:
                logger.warning(
                    "UniQL: %s relative error (%.4f) exceeded max_error (%.4f) at rank %d, bits %d.",
                    name,
                    best_err,
                    self.max_error,
                    best_rank,
                    best_bits,
                )
                if self.fallback_to_original:
                    logger.info("UniQL: Keeping original weights for %s.", name)
                    fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            if best_w_rec is None:
                fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # Reconstruct and assign weights to base_layer
            if is_conv1d:
                w_to_assign = best_w_rec.t().contiguous().to(dtype=orig_dtype, device=orig_device)
            else:
                w_to_assign = best_w_rec.to(dtype=orig_dtype, device=orig_device)

            if w_to_assign.shape != base_layer.weight.shape:
                logger.error("UniQL shape mismatch for %s: %s vs %s", name, w_to_assign.shape, base_layer.weight.shape)
                fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            base_layer.weight.data.copy_(w_to_assign)
            new_layer = base_layer
            compressed_layer = new_layer
            # Preserve existing LoRA adapter weights if the source module has them
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
               module.lora_A is not None and module.lora_B is not None:
                src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                src_lB = getattr(module.lora_B, 'weight', module.lora_B)
                if hasattr(new_layer, 'lora_A') and new_layer.lora_A is not None:
                    if hasattr(src_lA, 'data') and src_lA.shape == new_layer.lora_A.shape:
                        new_layer.lora_A.data.copy_(src_lA.data)
                if hasattr(new_layer, 'lora_B') and new_layer.lora_B is not None:
                    if hasattr(src_lB, 'data') and src_lB.shape == new_layer.lora_B.shape:
                        new_layer.lora_B.data.copy_(src_lB.data)

            if hasattr(module, 'base_layer'):
                module.base_layer = compressed_layer  # put compressed layer inside wrapper
                # module itself (LoRALinear wrapper) remains the replacement
                new_module = module
            else:
                new_module = compressed_layer

            if self.quantize_bias and getattr(base_layer, "bias", None) is not None and base_layer.bias is not None:
                b_q = self._quantize(base_layer.bias.data.float(), best_bits)
                base_layer.bias.data.copy_(b_q.to(dtype=orig_dtype, device=orig_device))

            self.compressed_layers.append(
                {
                    "name": name,
                    "original_shape": (in_f, out_f),
                    "rank": best_rank,
                    "bits": best_bits,
                    "rel_error": float(best_err),
                }
            )
            total_error += best_err

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (total_error / len(self.compressed_layers)) if self.compressed_layers else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": fallback_layers,
                "num_compressed_layers": len(self.compressed_layers),
                "num_fallback": len(fallback_layers),
                "rank": self.rank,
                "bits": self.bits,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
            }
        )
        logger.info(
            "UniQL compression finished: %d layers compressed, avg error: %.4f",
            len(self.compressed_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("uniql")
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

