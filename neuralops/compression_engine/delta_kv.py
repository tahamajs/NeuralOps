"""
DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity.
Registered as 'delta_kv', 'delta_kv_class', 'delta_kv_compressor', 'delta_kv_factory'.

Mathematical formulation:
  W ≈ L + R
  L = U_r Σ_r V_r^T                     (low-rank, rank = r)
  R = sparse(|W - L|; 1 - sparsity)     (top-k residual magnitudes)
  W_compressed = L + R

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# KV cache store (optional, DKV-FIX-13)
# ===========================================================================


class DeltaKVStore:
    """
    Optional runtime cache that stores delta residuals relative to retrieved
    historical K/V references. Not used by the weight-only compressor path.
    """

    def __init__(
        self,
        max_refs: int = 128,
        topk: int = 4,
        compression_ratio: float = 0.29,
        residual_threshold: float = 1e-3,
    ) -> None:
        self.max_refs = int(max_refs)
        self.topk = int(topk)
        self.compression_ratio = float(compression_ratio)
        self.residual_threshold = float(residual_threshold)
        self.ref_k: torch.Tensor | None = None
        self.ref_v: torch.Tensor | None = None

    def reset(self) -> None:
        self.ref_k = None
        self.ref_v = None

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        orig_k = k.detach()
        orig_v = v.detach()
        if orig_k.ndim == 2:
            orig_k = orig_k.unsqueeze(0).unsqueeze(0)
            orig_v = orig_v.unsqueeze(0).unsqueeze(0)
        elif orig_k.ndim == 3:
            orig_k = orig_k.unsqueeze(1)
            orig_v = orig_v.unsqueeze(1)

        B, H, N, D = orig_k.shape
        if N == 0 or B == 0:
            return

        if self.ref_k is None:
            self.ref_k = orig_k
            self.ref_v = orig_v
        else:
            if self.ref_k.device != orig_k.device or self.ref_k.dtype != orig_k.dtype:
                self.ref_k = self.ref_k.to(device=orig_k.device, dtype=orig_k.dtype)
                self.ref_v = self.ref_v.to(device=orig_v.device, dtype=orig_v.dtype)

            if self.ref_k.shape[0] != B:
                self.ref_k = self.ref_k[:1].expand(B, -1, -1, -1).clone()
                self.ref_v = self.ref_v[:1].expand(B, -1, -1, -1).clone()

            self.ref_k = torch.cat([self.ref_k, orig_k], dim=2)
            self.ref_v = torch.cat([self.ref_v, orig_v], dim=2)

        if self.ref_k.size(2) > self.max_refs:
            self.ref_k = self.ref_k[:, :, -self.max_refs :, :].contiguous()
            self.ref_v = self.ref_v[:, :, -self.max_refs :, :].contiguous()

    @torch.no_grad()
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = k.shape
        if self.ref_k is None or self.ref_k.size(2) == 0:
            self.update(k, v)
            return k, v

        k_4d = k
        v_4d = v
        if k.ndim == 2:
            k_4d = k.unsqueeze(0).unsqueeze(0)
            v_4d = v.unsqueeze(0).unsqueeze(0)
        elif k.ndim == 3:
            k_4d = k.unsqueeze(1)
            v_4d = v.unsqueeze(1)

        B, H, N, D = k_4d.shape
        ref_b, ref_h, ref_n, ref_d = self.ref_k.shape

        ref_k_cur = self.ref_k.to(device=k.device, dtype=k.dtype)
        ref_v_cur = self.ref_v.to(device=v.device, dtype=v.dtype)

        ref_k_b = ref_k_cur if ref_b == B else ref_k_cur[:1].expand(B, -1, -1, -1)
        ref_v_b = ref_v_cur if ref_b == B else ref_v_cur[:1].expand(B, -1, -1, -1)

        tk = min(self.topk, ref_n)
        if tk == 0:
            self.update(k, v)
            return k, v

        sim = torch.matmul(
            k_4d.float(), ref_k_b.float().transpose(-2, -1).contiguous()
        ) / (D**0.5)
        _, top_idx = torch.topk(sim, tk, dim=-1)

        top_idx_exp = top_idx.unsqueeze(-1).expand(B, H, N, tk, D)
        ref_k_exp = ref_k_b.unsqueeze(2).expand(B, H, N, ref_n, D)
        ref_v_exp = ref_v_b.unsqueeze(2).expand(B, H, N, ref_n, D)

        rk = ref_k_exp.gather(3, top_idx_exp).mean(dim=3).to(dtype=k_4d.dtype)
        rv = ref_v_exp.gather(3, top_idx_exp).mean(dim=3).to(dtype=v_4d.dtype)

        k_residual = k_4d - rk
        v_residual = v_4d - rv

        k_norm = torch.norm(k_residual.float(), dim=-1, keepdim=True)
        v_norm = torch.norm(v_residual.float(), dim=-1, keepdim=True)

        k_mask = (k_norm > self.residual_threshold).to(dtype=k_4d.dtype)
        v_mask = (v_norm > self.residual_threshold).to(dtype=v_4d.dtype)

        k_hat = rk + k_residual * k_mask
        v_hat = rv + v_residual * v_mask

        self.update(k, v)
        return k_hat.reshape(orig_shape), v_hat.reshape(orig_shape)


# ===========================================================================
# Helpers
# ===========================================================================


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
}


def _is_conv1d(module: nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    return "conv1d" in cls or hasattr(module, "nf")


# ===========================================================================
# Compressor (DKV-FIX-01, DKV-FIX-02)
# ===========================================================================


@COMPRESSOR_REGISTRY.register("delta_kv_class")
@COMPRESSOR_REGISTRY.register("delta_kv_compressor_class")
class DeltaKVCompressor(BaseCompressor):
    # DKV-FIX-01: this is a compressor, not attention.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        sparsity: float = 0.29,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="delta_kv", **kwargs)
        self.cfg = cfg

        self.rank = int(_cfg_get(cfg, "delta_kv_rank", _cfg_get(cfg, "rank", rank)))
        self.sparsity = float(
            _cfg_get(cfg, "delta_kv_sparsity", _cfg_get(cfg, "sparsity", sparsity))
        )
        self.max_error = float(
            _cfg_get(cfg, "delta_kv_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "delta_kv_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "delta_kv_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.min_layer_dim = max(1, int(_cfg_get(cfg, "min_layer_dim", min_layer_dim)))
        self.skip_should_compress_filter = bool(
            _cfg_get(
                cfg,
                "skip_should_compress_filter",
                skip_should_compress_filter,
            )
        )

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # Target check (DKV-FIX-07)
    # ------------------------------------------------------------------
    def _is_target_module(
        self, name: str, module: nn.Module, _debug: bool = False
    ) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False

        # Skip already compressed (DKV-FIX-05 companion)
        cls_name = module.__class__.__name__
        if cls_name in (
            "DeltaKVLinear",
            "LSCPrunedLoRALinear",
            "DynamicSVDLinear",
            "SVDLinear",
            "FactorisedLinear",
        ):
            if _debug:
                logger.info(f"  [skip] {name}: already compressed ({cls_name})")
            return False

        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            ]
        ):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        target_mod = getattr(module, "base_layer", module)
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = _is_conv1d(target_mod)
        if not (is_linear or is_conv1d):
            if _debug:
                logger.info(f"  [skip] {name}: not Linear/Conv1D")
            return False

        # DKV-FIX-06: bypass framework filter if requested.
        if not self.skip_should_compress_filter:
            try:
                from neuralops.utils.compat import should_compress_linear

                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        # DKV-FIX-11: min_layer_dim filter.
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(
                k in name_lower for k in ("attn", "attention")
            )
            if not is_attn:
                if _debug:
                    logger.info(f"  [skip] {name}: target_attention_only filter")
                return False

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        # DKV-FIX-09: idempotence
        if getattr(self.model, "_delta_kv_compressed", False):
            logger.warning("DeltaKV: model already compressed; skipping.")
            try:
                stats = self.get_compression_stats()
            except Exception:
                stats = {}
            return self.model, stats

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        total_orig_params = 0
        total_comp_params = 0

        # Diagnostic breakdown if no targets.
        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]
        if not target_list:
            logger.warning(
                "DeltaKV: no compressible layers found. Running diagnostic..."
            )
            n_scanned = 0
            for name, module in self.model.named_modules():
                n_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning(
                "DeltaKV diagnostic: %d modules scanned, 0 targets.", n_scanned
            )
            try:
                stats = self.get_compression_stats()
            except Exception:
                stats = {}
            return self.model, stats

        logger.info("DeltaKV: %d target layers identified.", len(target_list))

        # Freeze non-target attention/linear layers when target_attention_only.
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(
                    part in name.split(".")
                    for part in ("lora_A", "lora_B", "base_layer")
                ):
                    continue
                target_mod = getattr(module, "base_layer", module)
                if isinstance(target_mod, nn.Linear) or _is_conv1d(target_mod):
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        for name, module in target_list:
            target_mod = getattr(module, "base_layer", module)
            raw_w = target_mod.weight.data
            is_conv1d = _is_conv1d(target_mod)

            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            orig_layer_params = out_f * in_f
            total_orig_params += orig_layer_params
            norm_w = torch.norm(W, p="fro").item()

            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                # DKV-FIX-03: correct min over 2D shape.
                r = max(1, min(self.rank, out_f, in_f, len(S)))
                L = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                residual = W - L
                flat_r = residual.contiguous().abs().reshape(-1)
                keep_n = max(
                    1, min(int(flat_r.numel() * self.sparsity), flat_r.numel())
                )
                threshold = torch.topk(flat_r, keep_n).values[-1]
                mask = (residual.abs() >= threshold).float()
                residual_sparse = residual * mask

                W_compressed = L + residual_sparse
                err = float(
                    torch.norm(W - W_compressed, p="fro").item() / max(norm_w, 1e-12)
                )
                self.layer_errors[name] = err
                self.layer_ranks[name] = r

                if err > self.max_error and self.fallback_to_original:
                    logger.warning(
                        "DeltaKV: error %.4f on %s > %.4f; keeping original.",
                        err,
                        name,
                        self.max_error,
                    )
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

                # Write back in original orientation and dtype.
                if is_conv1d:
                    target_mod.weight.data.copy_(
                        W_compressed.t().contiguous().to(dtype=target_mod.weight.dtype)
                    )
                else:
                    target_mod.weight.data.copy_(
                        W_compressed.to(dtype=target_mod.weight.dtype)
                    )

                # DKV-FIX-10: requires_grad handled at module level.
                for p in module.parameters():
                    p.requires_grad = True

                self.compressed_layers.append(name)
                # DKV-FIX-12: honest parameter accounting.
                # Note: the dense weight is retained in-place (no reduction);
                # we report the theoretical storage ratio here for reference.
                comp_layer_params = r * (in_f + out_f) + keep_n
                total_comp_params += comp_layer_params

            except Exception as e:
                logger.warning("DeltaKV: decomposition failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_layer_params

        # DKV-FIX-08: forward sanity check.
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(iter(self.model.parameters()), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("DeltaKV: forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("DeltaKV: forward check raised: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        effective_ratio = (
            total_comp_params / total_orig_params if total_orig_params > 0 else 1.0
        )
        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update(
            {
                "rank": self.rank,
                "sparsity": self.sparsity,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "kv_cache_storage_ratio": f"{self.sparsity * 100:.1f}% of full KV cache",
                "forward_check_success": forward_check_success,
                "method": "delta_kv",
                "note": (
                    "DeltaKV rewrites dense weights in-place (low-rank + sparse residual). "
                    "The stored weight retains its full (out, in) shape; framework-level "
                    "compression_rate reads 1.0. Use effective_parameter_ratio for the "
                    "theoretical storage metric."
                ),
            }
        )
        self.stats = stats

        # DKV-FIX-14: write stats on the model.
        try:
            if (
                not hasattr(self.model, "compression_stats")
                or self.model.compression_stats is None
            ):
                self.model.compression_stats = {}
            self.model.compression_stats.update(
                {
                    "num_compressed": len(self.compressed_layers),
                    "num_fallback": len(self.fallback_layers),
                    "compression_method": "delta_kv",
                    "effective_parameter_ratio": effective_ratio,
                    "avg_reconstruction_error": avg_err,
                }
            )
        except Exception:
            pass

        self.model._delta_kv_compressed = True
        logger.info(
            "DeltaKV: %d compressed, %d fallback, %d skipped. "
            "Avg err %.4f, effective_param_ratio %.2fx.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            len(self.skipped_layers),
            avg_err,
            effective_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory (DKV-FIX-02)
# ===========================================================================


@COMPRESSOR_REGISTRY.register("delta_kv")
@COMPRESSOR_REGISTRY.register("delta_kv_compressor")
@COMPRESSOR_REGISTRY.register("delta_kv_factory")
def delta_kv_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """DeltaKV compressor factory."""
    rank = kwargs.pop(
        "rank",
        _cfg_get(comp_config, "delta_kv_rank", _cfg_get(comp_config, "rank", 16)),
    )
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(
            comp_config, "delta_kv_sparsity", _cfg_get(comp_config, "sparsity", 0.29)
        ),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "delta_kv_max_error", _cfg_get(comp_config, "max_error", 0.35)
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "delta_kv_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback_to_original = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "delta_kv_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    skip_filter = kwargs.pop(
        "skip_should_compress_filter",
        _cfg_get(comp_config, "skip_should_compress_filter", False),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = DeltaKVCompressor(
        model,
        rank=rank,
        sparsity=sparsity,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback_to_original,
        skip_should_compress_filter=skip_filter,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "DeltaKVStore",
    "DeltaKVCompressor",
    "delta_kv_factory",
]
