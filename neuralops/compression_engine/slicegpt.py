"""
PCA-ASVD: Per-Layer Activation-Aware Low-Rank Compression.

NOTE ON NAMING: This implementation is NOT SliceGPT (Microsoft, 2023).
- SliceGPT uses a GLOBAL rotation + reduces the model's hidden dimension.
- This implementation uses PER-LAYER PCA of activations + reduces each layer's rank.

The two methods share the idea of orthogonal rotation but differ fundamentally
in scope and mechanism.

=== PRODUCTION HARDENING (2026-10-04) ===
This version addresses three critical production issues:
  1. RAM Bomb: Online covariance accumulation replaces storing all activations.
     Memory: O(d_in²) per layer instead of O(N × seq_len × d_in).
  2. Hook Memory Leak: try/finally guarantees hook removal even on crash.
  3. Retained Graph: .detach().float() prevents VRAM leak from autograd graph.

Mathematical Formulation:
  1. Orthogonal Coordinate Invariance:
     W_l X_l = (W_l Q_l) (Q_l^T X_l)
  2. Eigen-decomposition of Activation Covariance:
     C = X^T X = Q Λ Q^T
  3. Coordinate Slicing + Low-Rank Factorization:
     W ≈ (W Q[:, :keep]) Q[:, :keep]^T
     Stored as U_factor @ Vt_factor with (d_out + d_in) × keep parameters.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import get_calibration_tensors
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

# Real low-rank factorization layer (used by AA-SVD / ERC-SVD)
try:
    from neuralops.compression_engine.aa_svd_compressor import SVDLinear
except Exception as _svd_import_err:  # pragma: no cover - defensive fallback
    SVDLinear = None
    _SVD_IMPORT_ERROR = str(_svd_import_err)
else:
    _SVD_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("pca_asvd")
@COMPRESSOR_REGISTRY.register("pca_asvd_compressor")
@COMPRESSOR_REGISTRY.register("activation_aware_svd")
# Backward-compat aliases (deprecated — will be removed)
@COMPRESSOR_REGISTRY.register("slicegpt")
@COMPRESSOR_REGISTRY.register("slicegpt_compressor")
class PCAActivationAwareSVDCompressor(BaseCompressor):
    """
    Per-Layer Activation-Aware Low-Rank Compression (PCA-ASVD).

    For each target linear layer:
      1. Collect input activation covariance C = E[X^T X] online (no RAM bomb).
      2. Compute orthogonal rotation Q via eigendecomposition of C.
      3. Rotate weight: W_pca = W @ Q.
      4. Truncate to top-`keep` principal coordinates.
      5. Store as rank-`keep` factorization: U = W_pca[:, :keep], V = Q[:, :keep]^T.

    Parameter reduction: (d_out + d_in) × keep instead of d_out × d_in.

    Memory: O(d_in²) per layer for covariance (not O(N × seq_len × d_in)).
    Stability: .detach() prevents autograd graph retention.
    Safety: try/finally guarantees hook cleanup on failure.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        sparsity: float = 0.6,
        non_uniform: bool = False,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        target_attention_only: bool = False,
        device: str | torch.device | None = None,
        max_error: float = 0.5,
        fallback_to_original: bool = True,
        target_modules: list[str] | None = None,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="pca_asvd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            sparsity = _cfg_get(cfg, "slicegpt_sparsity", _cfg_get(cfg, "sparsity", sparsity))
            non_uniform = _cfg_get(cfg, "non_uniform", non_uniform)
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            use_lora = _cfg_get(cfg, "use_lora", use_lora)
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])
        if "use_lora" in kwargs:
            use_lora = bool(kwargs["use_lora"])
        if "lora_r" in kwargs:
            lora_r = int(kwargs["lora_r"])

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.non_uniform = non_uniform
        self.target_attention_only = bool(target_attention_only)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.use_lora = bool(use_lora) or (lora_r > 0)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.stats: dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """
        Unified critical-layer protection delegated to should_compress_linear.
        """
        name_l = name.lower()
        if self.target_attention_only:
            attn_keywords = [
                "attn", "attention",
                "q_proj", "k_proj", "v_proj", "o_proj",
                "c_attn", "c_proj",
                "query", "key", "value",
            ]
            if not any(k in name_l for k in attn_keywords):
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
        # Skip layers that are already factorized
        if SVDLinear is not None and isinstance(base_mod, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    # FIX 1: Online covariance accumulation (no RAM bomb)
    # ------------------------------------------------------------------
    def _collect_layer_covariances(self) -> dict[str, torch.Tensor]:
        """
        Collect input activation covariances online, avoiding catastrophic RAM usage.

        Instead of storing all activations (which requires O(N × seq_len × d_in) RAM),
        we accumulate C = X^T X directly inside each hook. Memory is bounded by
        O(d_in²) per layer.

        FIX 2: hooks are removed in a `finally` block to prevent leaks on crash.
        """
        self.model.eval()
        covariances: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0]
                    if inp.numel() == 0:
                        return
                    # FIX 3: .detach() prevents autograd graph retention
                    inp = inp.detach().float()
                    # Flatten leading dims: (B, seq, d) -> (B*seq, d)
                    d_in = inp.shape[-1]
                    inp_flat = inp.reshape(-1, d_in)

                    # Online covariance: C += X^T X
                    batch_cov = torch.matmul(inp_flat.T, inp_flat).cpu()
                    batch_count = inp_flat.shape[0]

                    if name not in covariances:
                        covariances[name] = batch_cov
                        counts[name] = batch_count
                    else:
                        covariances[name] += batch_cov
                        counts[name] += batch_count
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

        # FIX 2: try/finally guarantees hook cleanup even on crash
        try:
            with torch.no_grad():
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
        except Exception as e:
            logger.warning(
                "Forward pass during calibration raised %s. "
                "Covariances may be incomplete.",
                e,
            )
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # Normalize covariances by sample count
        for name in list(covariances.keys()):
            covariances[name] /= max(1, counts[name])

        return covariances

    # ------------------------------------------------------------------
    # FIX: _compute_orthogonal_rotation takes covariance directly
    # ------------------------------------------------------------------
    def _compute_orthogonal_rotation(
        self,
        cov: torch.Tensor | None,
        d_in: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthogonal rotation Q via eigen-decomposition of pre-computed covariance.

        Args:
            cov: Pre-computed covariance matrix C = E[X^T X], shape (d_in, d_in).
                 If None or invalid, falls back to random orthogonal basis.
            d_in: Expected input dimension.
            device: Target device for output tensors.

        Returns:
            (Q, eigenvalues) where Q is orthogonal (d_in, d_in) and eigenvalues
            are sorted in descending order.
        """
        if device is None:
            device = cov.device if cov is not None else torch.device("cpu")

        if cov is not None and cov.numel() > 0:
            cov = cov.to(device)
            if cov.shape == (d_in, d_in):
                # Add small damping for numerical stability
                cov = cov + 1e-4 * torch.eye(d_in, device=device)
                try:
                    eigvals, Q = torch.linalg.eigh(cov)
                    # eigh returns ascending order; we want descending
                    idx = torch.argsort(eigvals, descending=True)
                    return Q[:, idx], eigvals[idx]
                except Exception as e:
                    logger.debug(
                        "eigh covariance failed (%s). Falling back to QR random basis.",
                        e,
                    )

        # Fallback: random orthogonal basis
        rand_mat = torch.randn(d_in, d_in, device=device)
        Q, _ = torch.linalg.qr(rand_mat)
        return Q, torch.ones(d_in, device=device)

    @staticmethod
    def _detect_lora(module: nn.Module) -> tuple[int, float]:
        """Detect existing LoRA rank and alpha from a wrapped module, if any."""
        if not hasattr(module, "lora_A") or module.lora_A is None:
            return 0, 1.0
        rank = 0
        la = getattr(module, "lora_A", None)
        if la is not None:
            if hasattr(la, "weight") and la.weight is not None and la.weight.dim() >= 1:
                rank = int(la.weight.shape[0])
            elif isinstance(la, nn.Parameter) and la.dim() >= 1:
                rank = int(la.shape[0])
        alpha = 1.0
        for attr in ("lora_alpha", "alpha"):
            a = getattr(module, attr, None)
            if a is not None:
                try:
                    alpha = float(a)
                except (TypeError, ValueError):
                    pass
                break
        return rank, alpha

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """
        Applies PCA rotation + rank-r truncation and replaces each weight
        with a real SVDLinear low-rank factorization.
        """
        if SVDLinear is None:
            logger.error(
                "PCA-ASVD: SVDLinear could not be imported (%s). Cannot perform "
                "real compression; returning model unchanged.",
                _SVD_IMPORT_ERROR,
            )
            stats = self.get_compression_stats()
            stats.update(
                {
                    "is_noop": True,
                    "noop_reason": "svd_linear_import_failed",
                    "method": "pca_asvd_unavailable",
                }
            )
            return self.model, stats

        self.layer_errors = {}
        self.layer_ranks = {}

        # FIX: Collect covariances, not activations
        covariances_map = self._collect_layer_covariances()

        sliced_layers_count = 0
        total_params_before = 0
        total_params_after = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_mod = getattr(module, "base_layer", module)
            raw_w = base_mod.weight.data
            cls_name = base_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            # FIX 3: .detach() prevents autograd graph retention
            if is_conv1d:
                W = raw_w.detach().t().float()
                d_out, d_in = W.shape
            else:
                W = raw_w.detach().float()
                d_out, d_in = W.shape

            total_params_before += d_out * d_in

            # FIX: pass covariance (not activations)
            cov = covariances_map.get(name, None)
            Q, _ = self._compute_orthogonal_rotation(cov, d_in, device=device)
            Q = Q.to(device=device, dtype=torch.float32)

            # Rotate weight into PCA basis
            W_pca = torch.matmul(W, Q)

            # Determine number of principal coordinates to retain
            if self.non_uniform and any(k in name for k in ("query", "key", "q_lin", "k_lin")):
                head_sparsity = max(0.0, self.sparsity * 0.75)
                keep = max(1, int(d_in * (1.0 - head_sparsity)))
            elif self.non_uniform and any(k in name for k in ("mlp", "ffn", "dense")):
                mlp_sparsity = min(0.9, self.sparsity * 1.15)
                keep = max(1, int(d_in * (1.0 - mlp_sparsity)))
            else:
                keep = max(1, int(d_in * (1.0 - self.sparsity)))

            # Rank r cannot exceed min(d_out, d_in)
            effective_keep = min(keep, d_out, d_in)

            # Reconstruct truncated dense weight for error measurement only
            W_trunc = W_pca.clone()
            if effective_keep < d_in:
                W_trunc[:, effective_keep:] = 0.0
            W_reconstructed = torch.matmul(W_trunc, Q.T)

            norm_orig = torch.norm(W, p="fro").clamp_min(1e-8)
            rel_err = (torch.norm(W - W_reconstructed, p="fro") / norm_orig).item()
            self.layer_errors[name] = float(rel_err)

            # Fallback on high reconstruction error
            if rel_err > self.max_error:
                logger.warning(
                    "PCA-ASVD reconstruction error %.4f on %s exceeded threshold %.4f%s",
                    rel_err, name, self.max_error,
                    ". Skipping (fallback to original)."
                    if self.fallback_to_original
                    else ". Proceeding with sliced weight.",
                )
                if self.fallback_to_original:
                    for p in module.parameters():
                        p.requires_grad = False
                    total_params_after += d_out * d_in
                    continue

            # === REAL LOW-RANK FACTORIZATION ===
            # W_reconstructed = W_pca[:, :keep] @ Q[:, :keep]^T
            U_factor = W_pca[:, :effective_keep].contiguous().to(dtype=dtype, device=device)
            Vt_factor = Q[:, :effective_keep].T.contiguous().to(dtype=dtype, device=device)

            # Parameter count of factorization
            factorized_params = effective_keep * (d_out + d_in)

            # Only apply if it actually reduces parameter count
            if factorized_params >= d_out * d_in:
                logger.info(
                    "PCA-ASVD: keep=%d on %s would not reduce params (%d >= %d). Keeping dense.",
                    effective_keep, name, factorized_params, d_out * d_in,
                )
                total_params_after += d_out * d_in
                continue

            # Detect existing LoRA adapters
            existing_lora_r, existing_lora_alpha = self._detect_lora(module)
            if self.use_lora and self.lora_r > 0 and existing_lora_r == 0:
                used_lora_r = self.lora_r
                used_lora_alpha = self.lora_alpha
            else:
                used_lora_r = existing_lora_r
                used_lora_alpha = existing_lora_alpha

            # Build SVDLinear factorization
            svd_layer = SVDLinear(
                in_features=d_in,
                out_features=d_out,
                rank=effective_keep,
                bias=(getattr(base_mod, "bias", None) is not None and base_mod.bias is not None),
                dtype=dtype,
                device=device,
                lora_r=used_lora_r,
                lora_alpha=used_lora_alpha,
            )
            svd_layer.set_low_rank_factors(U_factor, Vt_factor)

            # Copy bias
            if (
                getattr(base_mod, "bias", None) is not None
                and base_mod.bias is not None
                and getattr(svd_layer, "bias", None) is not None
            ):
                svd_layer.bias.data.copy_(base_mod.bias.data)

            # Copy existing LoRA adapters
            if existing_lora_r > 0 and getattr(svd_layer, "lora_A", None) is not None:
                try:
                    src_la = getattr(module.lora_A, "weight", module.lora_A)
                    src_lb = getattr(module.lora_B, "weight", module.lora_B)
                    la_target = svd_layer.lora_A
                    lb_target = svd_layer.lora_B
                    if (
                        hasattr(src_la, "data")
                        and la_target is not None
                        and src_la.shape == la_target.shape
                    ):
                        la_target.data.copy_(src_la.data)
                    if (
                        hasattr(src_lb, "data")
                        and lb_target is not None
                        and src_lb.shape == lb_target.shape
                    ):
                        lb_target.data.copy_(src_lb.data)
                except Exception as e_lora:
                    logger.debug("PCA-ASVD LoRA adapter copy skipped on %s: %s", name, e_lora)

            # Replace the module
            is_wrapped = hasattr(module, "base_layer")
            if is_wrapped:
                module.base_layer = svd_layer
            else:
                replace_module(self.model, name, svd_layer)

            self.layer_ranks[name] = effective_keep
            sliced_layers_count += 1
            total_params_after += factorized_params

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy = torch.randint(0, 100, (1, 8), device=self.target_device)
                self.model(dummy)
        except Exception as e:
            logger.warning("PCA-ASVD post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors.values()) / len(self.layer_errors)
            if self.layer_errors
            else 0.0
        )
        compression_rate = (
            total_params_before / max(1, total_params_after)
            if total_params_after > 0
            else 1.0
        )

        stats = self.get_compression_stats()
        stats.update(
            {
                "sliced_layers": sliced_layers_count,
                "sparsity_target": self.sparsity,
                "non_uniform": self.non_uniform,
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "target_attention_only": self.target_attention_only,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "total_params_before": total_params_before,
                "total_params_after": total_params_after,
                "compression_rate": round(compression_rate, 4),
                "avg_reconstruction_error": avg_err,
                "forward_check_success": forward_check_success,
                "is_noop": (total_params_after >= total_params_before),
                "noop_reason": (
                    "no_factorizable_layers_or_fallback_all"
                    if total_params_after >= total_params_before
                    else None
                ),
                "method": "pca_asvd_real_low_rank_factorization",
                "activations_memory_mode": "online_covariance",
            }
        )
        self.stats = stats

        logger.info(
            "PCA-ASVD: %d layers factorized, avg recon error %.4f, "
            "params %d -> %d (%.2fx compression)",
            sliced_layers_count, avg_err,
            total_params_before, total_params_after, compression_rate,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("pca_asvd_factory")
# Backward-compat alias (deprecated)
@COMPRESSOR_REGISTRY.register("slicegpt_factory")
def pca_asvd_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Factory function for PCA-ASVD compressor."""
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "sparsity", _cfg_get(comp_config, "sparsity", 0.60)),
    )
    non_uniform = kwargs.pop("non_uniform", _cfg_get(comp_config, "non_uniform", False))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.60))
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "fallback_to_original", True),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    tokenizer = kwargs.pop(
        "tokenizer",
        getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None),
    )

    compressor = PCAActivationAwareSVDCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        non_uniform=non_uniform,
        target_attention_only=target_attention_only,
        max_error=max_error,
        fallback_to_original=fallback,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "PCAActivationAwareSVDCompressor",
    "pca_asvd_factory",
]