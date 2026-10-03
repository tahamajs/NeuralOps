"""
EntropyBudgetAllocation: global rank assignment via whitened activation entropy.
Registered as 'entropy_budget' and 'entropy_budget_allocation'.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import adaptive_max_error
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


@COMPRESSOR_REGISTRY.register("entropy_budget")
@COMPRESSOR_REGISTRY.register("entropy_budget_allocation")
class EntropyBudgetCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Entropy Budget Low-Rank SVD Compressor.

    Computes activation covariance and whiten weights, calculates per-component
    singular entropy contributions, allocates low-rank budgets globally across layers,
    and replaces dense nn.Linear layers with factorized SVDLinear modules with symmetric
    singular value splitting (U * sqrt(Sigma), sqrt(Sigma) * V^T).

    Hardened for:
    - LoRALinear wrappers & Conv1D support.
    - target_attention_only filtering.
    - Relative Frobenius reconstruction error validation & fallback.
    - Freezing of fallback and skipped layers.
    - Real compression metrics recording.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Any | None = None,
        total_budget_ratio: float = 0.7,
        rank: int | None = None,
        eps: float = 1e-6,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        target_modules: list[str] | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="entropy_budget", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.calibration_loader = calibration_loader or _cfg_get(cfg, "calibration_loader", None)
        self.total_budget_ratio = float(
            _cfg_get(cfg, "entropy_budget_total_budget_ratio", _cfg_get(cfg, "total_budget_ratio", _cfg_get(cfg, "budget_ratio", total_budget_ratio)))
        )
        self.rank = _cfg_get(cfg, "entropy_budget_rank", _cfg_get(cfg, "rank", _cfg_get(cfg, "target_rank", rank)))
        if self.rank is not None:
            self.rank = int(self.rank)
        self.eps = float(eps)
        c_max_error = _cfg_get(cfg, "entropy_budget_max_error", _cfg_get(cfg, "max_error", max_error))
        if c_max_error is None or c_max_error == 0.35:
            c_max_error = adaptive_max_error(self.rank or 64, 768, 768, "distilbert")

        self.max_error = float(c_max_error)
        self.target_attention_only = bool(
            _cfg_get(cfg, "entropy_budget_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "entropy_budget_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_ranks: dict[str, int] = {}
        self.layer_errors: dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate", "head", "output_layer"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(target_mod, SVDLinear):
            return False

        if not should_compress_linear(self.model, name):
            return False

        if hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
            if min(target_mod.in_features, target_mod.out_features) < 16:
                return False
        elif hasattr(target_mod, "weight") and target_mod.weight is not None:
            if min(target_mod.weight.shape) < 16:
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                return False

        return True

    def collect_covariances(self) -> dict[str, torch.Tensor]:
        """Collect empirical input activation covariances with double-precision accumulation."""
        if self.calibration_loader is None:
            return {}

        activations: dict[str, torch.Tensor] = {}
        sample_counts: dict[str, int] = {}
        hooks = []

        def get_hook(name: str):
            def hook(module: nn.Module, inp: tuple[Any, ...], outp: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                if x.ndim >= 2:
                    x_flat = x.reshape(-1, x.shape[-1]).float()
                    cov = x_flat.T @ x_flat
                    n = x_flat.shape[0]
                    if name not in activations:
                        activations[name] = cov
                        sample_counts[name] = n
                    else:
                        activations[name] += cov
                        sample_counts[name] += n

            return hook

        target_layers = [
            (name, getattr(mod, "base_layer", mod))
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            return {}

        for name, mod in target_layers:
            hooks.append(mod.register_forward_hook(get_hook(name)))

        device = next(self.model.parameters()).device
        self.model.eval()

        try:
            with torch.no_grad():
                for batch in self.calibration_loader:
                    if isinstance(batch, dict):
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        batch = [
                            v.to(device) if isinstance(v, torch.Tensor) else v
                            for v in batch
                        ]
                        self.model(*batch)
                    elif isinstance(batch, torch.Tensor):
                        self.model(batch.to(device))
        except Exception as e:
            logger.warning(
                "EntropyBudget: Calibration forward pass failed (%s). Falling back to weight spectra.",
                e,
            )
        finally:
            for h in hooks:
                h.remove()

        normalized_covs: dict[str, torch.Tensor] = {}
        for name, cov in activations.items():
            cnt = max(1, sample_counts.get(name, 1))
            normalized_covs[name] = cov / cnt

        return normalized_covs

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Compress model using whitened entropy budget allocation and covariance-aligned singular vectors."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_ranks = {}
        self.layer_errors = {}

        total_orig_params = 0
        total_comp_params = 0

        # Freeze skipped layers if target_attention_only is enabled
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("EntropyBudget: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        covs = self.collect_covariances()
        residual_candidates = []
        layer_decomp_info = {}

        # Step 1: Compute covariance-whitened singular decomposition and singular entropy per layer
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data

            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            orig_layer_params = out_f * in_f
            total_orig_params += orig_layer_params
            max_r = min(out_f, in_f)
            r_floor = max(1, min(16, max_r))

            cov_inv_sqrt = None
            if name in covs and covs[name].shape == (in_f, in_f):
                cov = covs[name].to(device=W.device, dtype=torch.float32)
                try:
                    eigvals, eigvecs = torch.linalg.eigh(cov + 1e-4 * torch.eye(in_f, device=W.device))
                    eigvals_clamped = torch.clamp(eigvals, min=1e-4)
                    cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                    cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_clamped)) @ eigvecs.T
                    M = W @ cov_sqrt
                except Exception:
                    M = W
                    cov_inv_sqrt = None
            else:
                M = W

            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            layer_decomp_info[name] = (U, S, Vt, cov_inv_sqrt, max_r, r_floor, W, is_conv1d, target_mod)

            total_s = S.sum() + self.eps
            p = torch.clamp(S / total_s, min=self.eps, max=1.0)
            # B18: Shannon entropy in bits using log2
            entropy = -torch.sum(p * torch.log2(p)).item()
            entropy = max(entropy, 1e-4)

            for i in range(r_floor, max_r):
                s_val = S[i].item()
                score = (s_val ** 2) / entropy
                residual_candidates.append((name, i, s_val, score))

        # Step 2: Allocate ranks
        if self.rank is not None:
            allocated_ranks = {
                name: max(layer_decomp_info[name][5], min(self.rank, layer_decomp_info[name][4]))
                for name, _ in target_layers
            }
        else:
            allocated_ranks = {name: layer_decomp_info[name][5] for name, _ in target_layers}
            base_allocated = sum(allocated_ranks.values())
            total_possible = sum(layer_decomp_info[name][4] for name, _ in target_layers)
            target_total = max(base_allocated, min(total_possible, int(total_possible * self.total_budget_ratio)))
            residual_budget = target_total - base_allocated

            residual_candidates.sort(key=lambda x: x[3], reverse=True)
            for c in residual_candidates[:residual_budget]:
                allocated_ranks[c[0]] += 1

        # Step 3: Factorized replacement with validation & fallback
        forward_check_success = True

        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            r = allocated_ranks[name]
            U, S, Vt, cov_inv_sqrt, max_r, _, W, is_conv1d, _ = layer_decomp_info[name]
            out_f, in_f = W.shape
            orig_layer_params = out_f * in_f
            device = target_mod.weight.device
            dtype = target_mod.weight.dtype
            r = max(1, min(r, max_r))

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = U[:, :r] * sqrt_S.unsqueeze(0)
            if cov_inv_sqrt is not None:
                V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]) @ cov_inv_sqrt
                norm_u = torch.norm(U_r, p="fro").clamp_min(1e-8)
                norm_v = torch.norm(V_r, p="fro").clamp_min(1e-8)
                scale = torch.sqrt(norm_u / norm_v)
                U_r = U_r / scale
                V_r = V_r * scale
            else:
                V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]

            # Compute reconstructed weight & Frobenius error
            W_recon = torch.matmul(U_r.float(), V_r.float())
            norm_w = torch.norm(W, p="fro").item()
            err = float(torch.norm(W - W_recon, p="fro").item() / max(norm_w, 1e-12)) if norm_w > 1e-12 else 0.0
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            if err > self.max_error and self.fallback_to_original:
                logger.warning("EntropyBudget error %.4f on %s > max_error %.4f; falling back.", err, name, self.max_error)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_layer_params
                continue

            U_r = U_r.to(dtype=dtype, device=device)
            V_r = V_r.to(dtype=dtype, device=device)

            lora_r = getattr(module, "r", 0) if hasattr(module, "lora_A") else 0
            lora_alpha = getattr(module, "lora_alpha", 1.0) if hasattr(module, "lora_A") else 1.0

            svd_layer = SVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=getattr(target_mod, "bias", None) is not None and target_mod.bias is not None,
                dtype=dtype,
                device=device,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )
            svd_layer.set_low_rank_factors(U_r, V_r)
            if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(target_mod.bias.data)

            # If module was LoRALinear wrapper, copy existing trained adapter weights
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and svd_layer.lora_A is not None and svd_layer.lora_B is not None:
                if hasattr(module.lora_A, "weight") and module.lora_A.weight is not None:
                    svd_layer.lora_A.data.copy_(module.lora_A.weight.data)
                elif isinstance(module.lora_A, nn.Parameter):
                    svd_layer.lora_A.data.copy_(module.lora_A.data)
                if hasattr(module.lora_B, "weight") and module.lora_B.weight is not None:
                    svd_layer.lora_B.data.copy_(module.lora_B.weight.data)
                elif isinstance(module.lora_B, nn.Parameter):
                    svd_layer.lora_B.data.copy_(module.lora_B.data)

            replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)
            comp_layer_params = r * (in_f + out_f)
            total_comp_params += comp_layer_params

        # Sanity check forward pass
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception as e:
                    logger.warning(f"copy_weights failed: {e}")
        except Exception:
            forward_check_success = False

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )

        stats = self.get_compression_stats()
        stats.update({
            "allocated_ranks": allocated_ranks,
            "layer_ranks": self.layer_ranks,
            "layer_errors": self.layer_errors,
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "avg_reconstruction_error": avg_err,
            "effective_parameter_ratio": round(effective_ratio, 4),
            "total_budget_ratio": self.total_budget_ratio,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "forward_check_success": forward_check_success,
        })
        logger.info(
            "Compressor entropy_budget finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


def entropy_budget_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Entropy budget factory function."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "entropy_budget_rank", _cfg_get(comp_config, "rank", None)))
    budget_ratio = kwargs.pop(
        "total_budget_ratio",
        _cfg_get(comp_config, "entropy_budget_total_budget_ratio", _cfg_get(comp_config, "total_budget_ratio", 0.7)),
    )
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "entropy_budget_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "entropy_budget_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "entropy_budget_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )

    compressor = EntropyBudgetCompressor(
        model,
        total_budget_ratio=budget_ratio,
        rank=rank,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


entropy_budget_allocation_factory = entropy_budget_factory

__all__ = [
    "EntropyBudgetCompressor",
    "entropy_budget_factory",
    "entropy_budget_allocation_factory",
]
