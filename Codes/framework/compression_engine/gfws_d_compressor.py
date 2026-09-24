"""GFWSVD: Fisher-weighted / Covariance-weighted SVD. Registered as 'gfws' and 'gfws_d'."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.pruning_utils import get_calibration_tensors
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


logger = logging.getLogger(__name__)


class GFWSLoRALinear(nn.Module):
    """
    Linear layer containing Fisher/Covariance-whitened SVD compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        self.weight = nn.Parameter(
            weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> GFWSLoRALinear:
        bias = linear.bias.detach() if hasattr(linear, "bias") and linear.bias is not None else None
        in_features = getattr(linear, "in_features", weight.shape[1])
        out_features = getattr(linear, "out_features", weight.shape[0])
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(x_flat, self.weight, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


@COMPRESSOR_REGISTRY.register("gfws")
@COMPRESSOR_REGISTRY.register("gfws_d")
class GFWSCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Generalized Fisher / Covariance-Weighted SVD Compressor.
    Whitens weights using empirical input activation covariance square-root Cov(X)^{1/2},
    performs low-rank truncated SVD, and de-whitens to restore optimal compressed weights.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic fallback to standard weight SVD if covariance whitening fails or error > max_error.
    - Safe fallback to original weights if even standard SVD exceeds max_error.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - Full FP32 computation for numerical stability, casting back to original dtype only when assigning.
    - Trainable LoRA adapters (lora_A, lora_B) for parameter-efficient fine-tuning.
    - Per-layer logging and detailed compression statistics.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        damping: float = 0.01,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        target_attention_only: bool = False,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="gfws", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "gfws_rank", getattr(cfg, "rank", rank))
            damping = getattr(cfg, "gfws_damping", getattr(cfg, "damping", damping))
            max_error = getattr(cfg, "gfws_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "gfws_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = getattr(cfg, "lora_r", getattr(cfg, "lora_rank", lora_r))
            lora_alpha = getattr(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = getattr(cfg, "skip_attention_qkv", skip_attention_qkv)

        self.rank = int(rank)
        self.damping = float(damping)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = target_attention_only
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.used_covariance: Dict[str, bool] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(module, GFWSLoRALinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only and not any(k in name.lower() for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "c_attn", "c_proj"]):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _collect_activation_covariances(self) -> Dict[str, torch.Tensor]:
        """Collect empirical input activation covariances across target modules."""
        self.model.eval()
        running_cov: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str) -> Any:
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any) -> None:
                if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach().float()
                    cls_name = module.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                    if is_conv1d:
                        in_f = module.weight.shape[0]
                    else:
                        in_f = module.weight.shape[1]
                    if x.ndim >= 2:
                        flat_x = x.reshape(-1, in_f)
                    else:
                        flat_x = x.unsqueeze(0)
                    n_samples = flat_x.shape[0]
                    if n_samples > 0:
                        batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
                        if layer_name not in running_cov:
                            running_cov[layer_name] = batch_cov
                            sample_counts[layer_name] = n_samples
                        else:
                            running_cov[layer_name] += batch_cov
                            sample_counts[layer_name] += n_samples

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
                logger.debug("GFWS covariance collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        covariances: Dict[str, torch.Tensor] = {}
        for name, cov_sum in running_cov.items():
            n = float(max(sample_counts.get(name, 1), 1))
            covariances[name] = cov_sum / n

        return covariances

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress layers via activation covariance whitening + low-rank SVD, with standard SVD fallback."""
        from framework.core.model_utils import replace_module

        covariances = self._collect_activation_covariances()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.used_covariance = {}
        seen_weights: set[int] = set()

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        for name, module in target_layers:
            if not hasattr(module, "weight") or module.weight is None:
                continue

            w_id = id(module.weight)
            if w_id in seen_weights:
                for p in module.parameters(): p.requires_grad = False
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            target_mod = getattr(module, 'base_layer', module)
            raw_w = target_mod.weight.data

            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()  # [out_features, in_features]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()      # [out_features, in_features]

            r = max(1, min(self.rank, out_f, in_f))
            device = target_mod.weight.device
            dtype = target_mod.weight.dtype
            W_norm = torch.norm(W, p="fro").item()

            used_cov = False
            W_approx = None
            final_err = 1.0

            cov = covariances.get(name, None)
            if cov is not None and cov.shape == (in_f, in_f):
                try:
                    cov_dev = cov.to(device=device, dtype=torch.float32)
                    diag_mean = torch.diagonal(cov_dev).mean().item()
                    if not math.isfinite(diag_mean) or diag_mean <= 0.0:
                        diag_mean = 1.0
                    lambda_reg = max(float(self.damping) * diag_mean, 1e-6)
                    cov_reg = cov_dev + lambda_reg * torch.eye(in_f, device=device, dtype=torch.float32)

                    eigvals, eigvecs = torch.linalg.eigh(cov_reg)
                    eigvals_clamped = torch.clamp(eigvals, min=1e-6)

                    # Cov(X)^{1/2} and Cov(X)^{-1/2}
                    cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                    cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_clamped)) @ eigvecs.T

                    # Whiten weights: M = W * Cov(X)^{1/2}
                    M = torch.matmul(W, cov_sqrt)
                    U, S, Vt = torch.linalg.svd(M, full_matrices=False)

                    # Low-rank approximation of M: U_r * S_r * Vt_r
                    M_r = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                    # De-whiten: candidate = M_r * Cov(X)^{-1/2}
                    candidate = torch.matmul(M_r, cov_inv_sqrt)

                    if not (torch.isnan(candidate).any() or torch.isinf(candidate).any()):
                        err = float(torch.norm(W - candidate, p="fro").item() / max(W_norm, 1e-12))
                        if err <= self.max_error:
                            W_approx = candidate
                            final_err = err
                            used_cov = True
                        else:
                            logger.debug(
                                "GFWS covariance de-whitened error (%.4f) > max_error (%.4f) on %s. Trying SVD fallback.",
                                err,
                                self.max_error,
                                name,
                            )
                except Exception as e:
                    logger.debug("GFWS covariance decomposition exception on layer %s: %s", name, e)

            # Fallback to standard SVD if covariance whitening was not used or failed
            if W_approx is None:
                try:
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    candidate = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                    if not (torch.isnan(candidate).any() or torch.isinf(candidate).any()):
                        err = float(torch.norm(W - candidate, p="fro").item() / max(W_norm, 1e-12))
                        W_approx = candidate
                        final_err = err
                        used_cov = False
                except Exception as e:
                    logger.warning("GFWS standard SVD failed on %s: %s", name, e)
                    W_approx = None
                    final_err = 1.0

            self.layer_errors[name] = final_err
            self.layer_ranks[name] = r
            self.used_covariance[name] = used_cov

            if W_approx is None or final_err > self.max_error:
                logger.warning(
                    "GFWS reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping GFWS compression for %s (fallback to original).", name)
                    for p in module.parameters(): p.requires_grad = False
                    self.fallback_layers.append(name)
                    continue

            # Replace with GFWSLoRALinear to support optional post-compression LoRA adapters
            new_layer = GFWSLoRALinear.from_linear(
                target_mod,
                weight=W_approx.to(dtype=dtype, device=device),
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
            )
            if hasattr(module, 'base_layer'):
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and new_layer.lora_A is not None and new_layer.lora_B is not None:
                    with torch.no_grad():
                        src_A = module.lora_A.weight.data if hasattr(module.lora_A, 'weight') else module.lora_A.data
                        src_B = module.lora_B.weight.data if hasattr(module.lora_B, 'weight') else module.lora_B.data
                        if src_A.shape == new_layer.lora_A.shape and src_B.shape == new_layer.lora_B.shape:
                            new_layer.lora_A.copy_(src_A.to(new_layer.lora_A.device, dtype=new_layer.lora_A.dtype))
                            new_layer.lora_B.copy_(src_B.to(new_layer.lora_B.device, dtype=new_layer.lora_B.dtype))
                module.base_layer = new_layer
                new_layer = module
            replace_module(self.model, name, new_layer)
            self.compressed_layers.append(name)

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, GFWSLoRALinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("GFWS post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update({
            "rank": self.rank,
            "damping": self.damping,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "used_covariance": self.used_covariance,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "method": "covariance_weighted_svd",
        })
        logger.info(
            "GFWS compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("gfws_factory")
@COMPRESSOR_REGISTRY.register("gfws_d_factory")
def gfws_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """GFWS factory."""
    rank = kwargs.pop("rank", getattr(comp_config, "gfws_rank", getattr(comp_config, "rank", 64)) if comp_config else 64)
    damping = kwargs.pop("damping", getattr(comp_config, "gfws_damping", getattr(comp_config, "damping", 0.01)) if comp_config else 0.01)
    max_error = kwargs.pop("max_error", getattr(comp_config, "gfws_max_error", getattr(comp_config, "max_error", 0.20)) if comp_config else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "gfws_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ) if comp_config else True)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = GFWSCompressor(
        model,
        rank=rank,
        damping=damping,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = ["GFWSCompressor", "GFWSLoRALinear", "gfws_factory"]
