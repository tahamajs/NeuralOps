"""
SliceGPT: Compress Large Language Models by Deleting Rows and Columns.
Registered in COMPRESSOR_REGISTRY as 'slicegpt' and 'slicegpt_compressor'.

Mathematical Formulation:
1. Orthogonal Coordinate Invariance:
   W_l X_l = (W_l Q_l) (Q_l^T X_l)
2. Eigen-decomposition of Activation Covariance:
   C = X X^T = Q Λ Q^T
3. Coordinate Slicing Transformation:
   W_sliced = W Q P_r,   P_r = [I_r, 0]^T

=== REAL COMPRESSION (2026-09-25 FIX) ===
This version ACTUALLY reduces parameter count by replacing the sliced dense
weight with an SVDLinear low-rank factorization:

    W_reconstructed = W_pca[:, :keep] @ V[:, :keep].T
                    = U_factor @ Vt_factor     where:
                      U_factor  = W_pca[:, :keep]     shape (d_out, keep)
                      Vt_factor = V[:, :keep].T       shape (keep, d_in)

The resulting parameter count is (d_out + d_in) * keep instead of
d_out * d_in, producing real compression when keep < d_out * d_in / (d_out + d_in).

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

# Real low-rank factorization layer (already used by AA-SVD / ERC-SVD)
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


@COMPRESSOR_REGISTRY.register("slicegpt")
@COMPRESSOR_REGISTRY.register("slicegpt_compressor")
class SliceGPTCompressor(BaseCompressor):
    """
    SliceGPT Compression Strategy (real low-rank implementation).

    Performs orthogonal rotation of each target weight matrix using the
    eigendecomposition of its input activation covariance, then truncates
    to the top-r principal coordinates. The truncated weight is stored as a
    rank-r SVDLinear factorization, achieving genuine parameter reduction.
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
        super().__init__(model, name="slicegpt", **kwargs)
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
            ]
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear) or "conv1d" in cls_name or hasattr(base_mod, "nf")
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

    def _collect_layer_activations(self) -> dict[str, torch.Tensor]:
        """Collect input activations across target linear layers."""
        self.model.eval()
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    activations.setdefault(name, []).append(inp.cpu())

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
                logger.debug(
                    "Forward pass for activation collection had exception: %s. "
                    "Using synthetic linear inputs.",
                    e,
                )

        for h in hooks:
            h.remove()

        aggregated: dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)
        return aggregated

    def _compute_orthogonal_rotation(
        self,
        X: torch.Tensor | None,
        d_in: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute orthogonal rotation matrix Q via eigen-decomposition of activation covariance."""
        if device is None:
            device = X.device if X is not None else torch.device("cpu")
        if X is not None and X.numel() > 0:
            X_flat = X.reshape(-1, X.shape[-1]).float().to(device)
            if X_flat.shape[-1] == d_in and X_flat.shape[0] > 1:
                cov = torch.matmul(X_flat.T, X_flat) / X_flat.shape[0]
                cov = cov + 1e-4 * torch.eye(d_in, device=device)
                try:
                    eigvals, Q = torch.linalg.eigh(cov)
                    idx = torch.argsort(eigvals, descending=True)
                    return Q[:, idx], eigvals[idx]
                except Exception as e:
                    logger.debug(
                        "eigh covariance failed (%s). Falling back to QR random basis.",
                        e,
                    )

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
        Applies SliceGPT rotation + rank-r truncation and replaces each weight
        with a real SVDLinear low-rank factorization.
        """
        if SVDLinear is None:
            logger.error(
                "SliceGPT: SVDLinear could not be imported (%s). Cannot perform "
                "real compression; returning model unchanged.",
                _SVD_IMPORT_ERROR,
            )
            stats = self.get_compression_stats()
            stats.update(
                {
                    "is_noop": True,
                    "noop_reason": "svd_linear_import_failed",
                    "method": "slicegpt_unavailable",
                }
            )
            return self.model, stats

        self.layer_errors = {}
        self.layer_ranks = {}
        activations_map = self._collect_layer_activations()
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

            if is_conv1d:
                W = raw_w.t().float()
                d_out, d_in = W.shape
            else:
                W = raw_w.float()
                d_out, d_in = W.shape

            total_params_before += d_out * d_in

            act = activations_map.get(name, None)
            Q, _ = self._compute_orthogonal_rotation(act, d_in, device=device)
            Q = Q.to(device=device, dtype=torch.float32)

            # Rotate weight into the PCA basis
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
                    "SliceGPT reconstruction error %.4f on %s exceeded threshold %.4f%s",
                    rel_err,
                    name,
                    self.max_error,
                    (
                        ". Skipping (fallback to original)."
                        if self.fallback_to_original
                        else ". Proceeding with sliced weight."
                    ),
                )
                if self.fallback_to_original:
                    logger.warning(
                        "Fallback triggered for %s. Layer is kept original and NOT wrapped in LoRA. "
                        "It will not be fine-tuned.",
                        name,
                    )
                    for p in module.parameters():
                        p.requires_grad = False
                    total_params_after += d_out * d_in
                    continue

            # === REAL LOW-RANK FACTORIZATION ===
            # W_reconstructed = W_pca[:, :keep] @ Q[:, :keep]^T
            # Factor:
            #   U_factor  = W_pca[:, :keep]           shape (d_out, keep)
            #   Vt_factor = Q[:, :keep].T             shape (keep, d_in)
            U_factor = W_pca[:, :effective_keep].contiguous().to(dtype=dtype, device=device)
            Vt_factor = Q[:, :effective_keep].T.contiguous().to(dtype=dtype, device=device)

            # Parameter count of the SVDLinear factorization
            factorized_params = effective_keep * (d_out + d_in)
            # Only apply if it actually reduces parameter count
            if factorized_params >= d_out * d_in:
                logger.info(
                    "SliceGPT: keep=%d on %s would not reduce params (%d >= %d). Keeping dense.",
                    effective_keep,
                    name,
                    factorized_params,
                    d_out * d_in,
                )
                total_params_after += d_out * d_in
                continue

            # Detect existing LoRA adapters so we can preserve their weights
            existing_lora_r, existing_lora_alpha = self._detect_lora(module)
            if self.use_lora and self.lora_r > 0 and existing_lora_r == 0:
                # Apply new LoRA on top of the slicegpt factorization
                used_lora_r = self.lora_r
                used_lora_alpha = self.lora_alpha
            else:
                used_lora_r = existing_lora_r
                used_lora_alpha = existing_lora_alpha

            # Build the SVDLinear factorization
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

            # Copy existing LoRA adapters if they were present on the source module
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
                    logger.debug("SliceGPT LoRA adapter copy skipped on %s: %s", name, e_lora)

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
            logger.warning("SliceGPT post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors.values()) / len(self.layer_errors) if self.layer_errors else 0.0
        )
        compression_rate = (
            total_params_before / max(1, total_params_after) if total_params_after > 0 else 1.0
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
                "method": "slicegpt_real_low_rank_factorization",
            }
        )
        self.stats = stats

        logger.info(
            "SliceGPT (real): %d layers factorized, avg recon error %.4f, "
            "params %d -> %d (%.2fx compression)",
            sliced_layers_count,
            avg_err,
            total_params_before,
            total_params_after,
            compression_rate,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("slicegpt_factory")
def slicegpt_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Factory function for SliceGPT compressor."""
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "slicegpt_sparsity", _cfg_get(comp_config, "sparsity", 0.60)),
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

    compressor = SliceGPTCompressor(
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
    "SliceGPTCompressor",
    "slicegpt_factory",
]
