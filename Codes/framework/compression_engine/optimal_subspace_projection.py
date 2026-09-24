"""
Optimal Subspace Projection: Activation-Aware Subspace Projection via Randomized SVD.
Registered in COMPRESSOR_REGISTRY as 'optimal_subspace_projection'.

Mathematical Formulation:
1. Activation Covariance:
   $$C = \\frac{1}{N} X^\\top X + \\epsilon I$$
2. Cholesky Preconditioning Factor:
   $$C = L L^\\top \\implies M = W L$$
3. Randomized Subspace Iteration:
   $$Y = M \\Omega, \\quad Y \\leftarrow M (M^\\top Y), \\quad Q = \\operatorname{qr}(Y)$$
4. Low-Rank Projection & Inversion:
   $$B = Q^\\top M = U \\Sigma V^\\top$$
   $$W_{osp} = (Q U_r \\Sigma_r V_r^\\top) L^{-1}$$
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import get_calibration_tensors
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class OptimalSubspaceLoRALinear(nn.Module):
    """
    Linear layer containing optimal subspace projected base weights (frozen)
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
        self.nx = in_features   # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        # Normalize to nn.Linear orientation (out_features, in_features); some
        # call sites pass Conv1D-style (in, out) weights for non-square layers.
        if weight.shape[0] == in_features and weight.shape[1] == out_features and in_features != out_features:
            weight = weight.t().contiguous()

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
    ) -> OptimalSubspaceLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        bias = base_layer.bias.detach() if hasattr(base_layer, "bias") and base_layer.bias is not None else None
        in_features = getattr(base_layer, "in_features", getattr(base_layer, "nx", None))
        out_features = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
        if in_features is None or out_features is None:
            out_features = out_features or weight.shape[0]
            in_features = in_features or weight.shape[1]

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

        # Preserve existing LoRA weights if present
        if hasattr(linear, "lora_A") and hasattr(linear, "lora_B") and layer.lora_A is not None and layer.lora_B is not None:
            with torch.no_grad():
                src_A = linear.lora_A.weight.data if hasattr(linear.lora_A, "weight") else linear.lora_A.data
                src_B = linear.lora_B.weight.data if hasattr(linear.lora_B, "weight") else linear.lora_B.data
                if src_A.shape == layer.lora_A.shape and src_B.shape == layer.lora_B.shape:
                    layer.lora_A.copy_(src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype))
                    layer.lora_B.copy_(src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype))

        if layer.lora_A is not None:
            layer.lora_A.requires_grad = True
        if layer.lora_B is not None:
            layer.lora_B.requires_grad = True

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)
        return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin", "self_attn"
}


@COMPRESSOR_REGISTRY.register("optimal_subspace_projection")
class OptimalSubspaceProjectionCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Optimal Subspace Projection Compressor using activation-aware randomized SVD.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.50).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection by class name and nf attribute without unreliable shape heuristics.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping to preserve key-value cache dimension alignment.
    - Tied / shared weight protection tracking parameter IDs.
    - Mathematical preconditioning inversion fix (L_inv vs L_inv.T).
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        rank_ratio: float = 0.5,
        power_iters: int = 2,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        min_rank: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="optimal_subspace_projection", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank_ratio = _cfg_get(cfg, "optimal_subspace_rank_ratio", _cfg_get(cfg, "rank_ratio", rank_ratio))
            power_iters = _cfg_get(cfg, "optimal_subspace_power_iters", _cfg_get(cfg, "power_iters", power_iters))
            max_error = _cfg_get(cfg, "optimal_subspace_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "optimal_subspace_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.calibration_loader = calibration_loader
        self.rank_ratio = max(0.01, min(float(rank_ratio), 1.0))
        self.power_iters = max(0, int(power_iters))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_texts = calibration_texts
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.min_rank = max(1, int(min_rank))

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
        self.stats: Dict[str, Any] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        """Check if layer is an attention QKV projection in a model with GQA."""
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
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only:
            is_attn = any(k in name.lower() for k in _ATTN_NAMES)
            if not is_attn:
                return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target model using optimal subspace projection."""
        act_covs: Dict[int, torch.Tensor] = {}
        hooks = []

        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                x = inputs[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                cov = torch.matmul(x_flat.T, x_flat) / max(x_flat.shape[0], 1)
                mod_id = id(module)
                if mod_id not in act_covs:
                    act_covs[mod_id] = cov.cpu()
                else:
                    act_covs[mod_id] += cov.cpu()

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(hook_fn))

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
                logger.debug("OptimalSubspaceProjection activation collection exception: %s", e)

        for h in hooks:
            h.remove()

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        seen_weights: Set[int] = set()
        total_orig_params = 0
        total_comp_params = 0

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                base_module = getattr(module, "base_layer", module)
                if not hasattr(base_module, "weight") or base_module.weight is None:
                    continue

                w_id = id(base_module.weight)
                if w_id in seen_weights:
                    logger.info("Skipping %s due to shared/tied weight.", name)
                    self.skipped_layers.append(name)
                    continue
                seen_weights.add(w_id)

                for p in module.parameters():
                    total_orig_params += p.numel()

                raw_w = base_module.weight.data
                cls_name = base_module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(base_module, "nf")

                W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
                d_out, d_in = W.shape
                device = raw_w.device
                dtype = raw_w.dtype

                C = act_covs.get(id(module), None)
                if C is not None and C.shape[-1] == d_in:
                    C = C.to(device)
                    diag_mean = torch.diagonal(C).mean().item()
                    reg = max(1e-4 * diag_mean, 1e-5)
                    C_reg = C + reg * torch.eye(d_in, device=device)
                    try:
                        L = torch.linalg.cholesky(C_reg)
                        L_inv = torch.linalg.inv(L)
                        if torch.isnan(L).any() or torch.isnan(L_inv).any():
                            raise ValueError("Cholesky NaNs")
                    except Exception:
                        L = torch.eye(d_in, device=device)
                        L_inv = torch.eye(d_in, device=device)
                else:
                    L = torch.eye(d_in, device=device)
                    L_inv = torch.eye(d_in, device=device)

                M = torch.matmul(W, L)
                rank = max(min(self.min_rank, min(d_out, d_in)), min(int(min(d_out, d_in) * self.rank_ratio), min(d_out, d_in)))
                Omega = torch.randn(M.shape[1], rank, device=device)
                Y = torch.matmul(M, Omega)
                for _ in range(self.power_iters):
                    Y = torch.matmul(M, torch.matmul(M.T, Y))

                try:
                    Q, _ = torch.linalg.qr(Y)
                    B = torch.matmul(Q.T, M)
                    U, S, Vt = torch.linalg.svd(B, full_matrices=False)
                    r = min(rank, len(S))
                    U_r = U[:, :r]
                    S_r = S[:r]
                    Vt_r = Vt[:r, :]
                    new_U = torch.matmul(Q, U_r)
                    # Correct de-conditioning: (Q @ U_r @ S_r @ Vt_r) @ L_inv
                    new_V = torch.matmul(Vt_r, L_inv)
                    new_weight = torch.matmul(new_U * S_r.unsqueeze(0), new_V)
                    new_weight = torch.nan_to_num(new_weight, nan=0.0)

                    w_norm = torch.norm(W, p="fro").item()
                    err = (torch.norm(W - new_weight, p="fro") / max(w_norm, 1e-12)).item()
                    self.layer_errors[name] = float(err)
                    self.layer_ranks[name] = r

                    if not math.isfinite(err) or err > self.max_error:
                        logger.warning(
                            "OptimalSubspaceProjection reconstruction error %.4f on %s exceeded threshold %.4f.",
                            err,
                            name,
                            self.max_error,
                        )
                        if self.fallback_to_original:
                            logger.info("Skipping OptimalSubspaceProjection for %s (fallback to original).", name)
                            self.fallback_layers.append(name)
                            for p in module.parameters():
                                p.requires_grad = False
                                total_comp_params += p.numel()
                            continue

                    W_target = new_weight
                    compressed_layer = OptimalSubspaceLoRALinear.from_linear(
                        module,
                        weight=W_target.to(dtype=dtype, device=device),
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    replace_module(self.model, name, compressed_layer)
                    self.compressed_layers.append(name)
                    for p in compressed_layer.parameters():
                        total_comp_params += p.numel()
                except Exception as e:
                    logger.warning("Subspace projection SVD exception in %s: %s", name, e)
                    self.fallback_layers.append(name)
                    self.layer_errors[name] = 1.0
                    for p in module.parameters():
                        p.requires_grad = False
                        total_comp_params += p.numel()

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    self.model(dummy.float())
        except Exception as e:
            logger.warning("OptimalSubspaceProjection post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        eff_ratio = total_comp_params / max(1, total_orig_params)
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "rank_ratio": self.rank_ratio,
            "power_iters": self.power_iters,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "effective_parameter_ratio": eff_ratio,
            "forward_check_success": forward_check_success,
            "method": "optimal_subspace_projection",
        })
        self.stats = stats
        return self.model, stats


# Alias
SubspaceProjectionCompressor = OptimalSubspaceProjectionCompressor


@COMPRESSOR_REGISTRY.register("optimal_subspace_projection_factory")
def optimal_subspace_projection_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """Optimal Subspace Projection factory."""
    try:
        calibration_loader = kwargs.pop("calibration_loader", _cfg_get(comp_config, "calibration_loader", None))
        rank_ratio = kwargs.pop("rank_ratio", _cfg_get(comp_config, "optimal_subspace_rank_ratio", _cfg_get(comp_config, "rank_ratio", 0.5)))
        power_iters = kwargs.pop("power_iters", _cfg_get(comp_config, "optimal_subspace_power_iters", _cfg_get(comp_config, "power_iters", 2)))
        max_error = kwargs.pop("max_error", _cfg_get(comp_config, "optimal_subspace_max_error", _cfg_get(comp_config, "max_error", 0.50)))
        fallback = kwargs.pop("fallback_to_original", _cfg_get(
            comp_config, "optimal_subspace_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
        ))
        target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
        tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None))
        lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
        lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
        return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

        compressor = OptimalSubspaceProjectionCompressor(
            model,
            calibration_loader=calibration_loader,
            rank_ratio=rank_ratio,
            power_iters=power_iters,
            tokenizer=tokenizer,
            max_error=max_error,
            fallback_to_original=fallback,
            target_attention_only=target_attention_only,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            comp_config=comp_config,
            **kwargs,
        )
        compressed_model, stats = compressor.compress()
        if return_stats:
            return compressed_model, stats
        return compressed_model
    except Exception as e:
        logger.error("optimal_subspace_projection_factory failed: %s", e)
        if kwargs.get("return_stats", False) or kwargs.get("return_tuple", False) or ('return_stats' in locals() and return_stats):
            return model, {}
        return model


@COMPRESSOR_REGISTRY.register("optimal_subspace_projection_inject")
@COMPRESSOR_REGISTRY.register("inject_optimal_subspace_projection")
def inject_optimal_subspace_projection(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook for Optimal Subspace Projection."""
    target = layer if layer is not None else model
    if target is None:
        return None
    try:
        return optimal_subspace_projection_factory(target, **kwargs)
    except Exception as e:
        logger.error("inject_optimal_subspace_projection failed: %s", e)
        return None


__all__ = [
    "OptimalSubspaceLoRALinear",
    "OptimalSubspaceProjectionCompressor",
    "SubspaceProjectionCompressor",
    "optimal_subspace_projection_factory",
    "inject_optimal_subspace_projection",
]
