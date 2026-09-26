"""
ProxSparse: Proximal Gradient Regularized Structured & Semi-Structured Sparsity.

Registered in COMPRESSOR_REGISTRY as 'proxsparse', 'proxsparse_pruner',
'proxsparse_compressor', 'proxsparse_factory', ...

Mathematical Formulation:
    min_W  0.5 * ||W - W_0||_F^2 + lambda * ||W||_1 + I_S(W)

where:
  - W_0 is the original (pretrained) weight,
  - ||.||_1 is the L1 regularizer,
  - I_S(W) is the indicator of the sparse constraint set S (percentile,
    block, or 2:4 semi-structured).

Iterative Proximal Gradient (ISTA with projection):
    for t = 1, ..., steps:
        grad_W    = W^(t) - W_0                  # gradient of smooth part
        W_grad    = W^(t) - eta * grad_W         # gradient descent step
        W_prox    = S_{eta*lambda}(W_grad)       # soft-thresholding
        W^(t+1)   = Pi_S(W_prox)                 # projection to S

where:
  - S_a(x) = sign(x) * max(|x| - a, 0)  (soft-thresholding)
  - Pi_S is the hard projection (percentile/2:4/block top-k)

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
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
)
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# Pruned LoRA linear layer
# ===========================================================================
class ProxSparsePrunedLoRALinear(nn.Module):
    """
    Frozen pruned base weight + trainable LoRA adapters.

    NOTE: `weight` is the ALREADY-PRUNED weight matrix. Forward does not
    re-apply the mask. `mask` is kept only as a compact bool buffer for
    downstream analysis / merge operations.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        pruned_weight: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else pruned_weight.device
        target_dtype = dtype if dtype is not None else pruned_weight.dtype

        # PS-FIX-02: weight is already masked; do not multiply again.
        self.weight = nn.Parameter(
            pruned_weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        # PS-FIX-03: bool buffer (1 byte per element, non-persistent).
        self.register_buffer(
            "mask",
            mask.bool().to(device=target_device),
            persistent=False,
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
                torch.empty((self.lora_r, in_features),
                            dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r),
                            dtype=torch.float32, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        pruned_weight: torch.Tensor,
        mask: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "ProxSparsePrunedLoRALinear":
        base_mod = getattr(linear, "base_layer", linear)

        # PS-FIX-09: robust in/out detection for Linear and Conv1D.
        cls_name = base_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")

        if is_conv1d:
            in_features = int(getattr(base_mod, "nx", pruned_weight.shape[1]))
            out_features = int(getattr(base_mod, "nf", pruned_weight.shape[0]))
        else:
            in_features = int(getattr(base_mod, "in_features", pruned_weight.shape[1]))
            out_features = int(getattr(base_mod, "out_features", pruned_weight.shape[0]))

        bias = None
        if getattr(base_mod, "bias", None) is not None:
            bias = base_mod.bias.detach()

        instance = cls(
            in_features=in_features,
            out_features=out_features,
            pruned_weight=pruned_weight,
            mask=mask,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=pruned_weight.device,
            dtype=pruned_weight.dtype,
        )

        # Preserve existing source LoRA parameters if shapes match.
        if hasattr(linear, "lora_A") and hasattr(linear, "lora_B"):
            src_A = getattr(linear.lora_A, "weight", linear.lora_A)
            src_B = getattr(linear.lora_B, "weight", linear.lora_B)
            if (src_A is not None and src_B is not None
                    and instance.lora_A is not None
                    and instance.lora_B is not None):
                if src_A.shape == instance.lora_A.shape:
                    instance.lora_A.data.copy_(src_A.data)
                if src_B.shape == instance.lora_B.shape:
                    instance.lora_B.data.copy_(src_B.data)
        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PS-FIX-02: no re-multiplication by mask; weight is already masked.
        w = self.weight.to(dtype=x.dtype, device=x.device)
        # PS-FIX-06: bias cast includes device.
        b = None
        if self.bias is not None:
            b = self.bias.to(dtype=x.dtype, device=x.device)

        out = F.linear(x, w, b)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
            out = out + lora_delta.to(out.dtype)
        return out


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("proxsparse")
@COMPRESSOR_REGISTRY.register("proxsparse_pruner")
@COMPRESSOR_REGISTRY.register("proxsparse_compressor")
class ProxSparsePruner(BaseCompressor):
    """
    ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile
    constraint, optional LoRA fine-tuning, and safe fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        lambda_prox: float = 0.01,
        steps: int = 5,
        learning_rate: float = 0.5,   # PS-FIX-04
        structured_24: bool = False,
        block_size: Tuple[int, int] = (1, 1),
        skip_attention_qkv: bool = True,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: Optional[float] = None,
        fallback_to_original: bool = True,
        min_numel: int = 16,
        target_modules: Optional[List[str]] = None,
        target_attention_only: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="proxsparse", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config", None)

        self.sparsity = max(
            0.0,
            min(
                float(sparsity if sparsity is not None
                      else _cfg_get(cfg_obj, "proxsparse_sparsity",
                                    _cfg_get(cfg_obj, "sparsity", 0.5))),
                0.95,
            ),
        )
        self.lambda_prox = max(
            0.0,
            float(lambda_prox if lambda_prox is not None
                  else _cfg_get(cfg_obj, "proxsparse_lambda",
                                _cfg_get(cfg_obj, "lambda_prox", 0.01))),
        )
        self.steps = max(
            1,
            int(steps if steps is not None
                else _cfg_get(cfg_obj, "proxsparse_steps",
                              _cfg_get(cfg_obj, "steps", 5))),
        )
        # PS-FIX-04
        self.learning_rate = max(
            1e-4,
            float(learning_rate if learning_rate is not None
                  else _cfg_get(cfg_obj, "proxsparse_lr",
                                _cfg_get(cfg_obj, "learning_rate", 0.5))),
        )
        self.structured_24 = bool(
            structured_24 if structured_24 is not None
            else _cfg_get(cfg_obj, "proxsparse_structured_24",
                          _cfg_get(cfg_obj, "structured_24", False))
        )
        self.block_size = block_size or _cfg_get(cfg_obj, "block_size", (1, 1))
        self.target_attention_only = bool(
            target_attention_only or _cfg_get(cfg_obj, "target_attention_only", False)
        )
        self.skip_attention_qkv = bool(
            skip_attention_qkv if skip_attention_qkv is not None
            else _cfg_get(cfg_obj, "skip_attention_qkv", True)
        ) and not self.target_attention_only
        self.use_lora = bool(
            use_lora or _cfg_get(cfg_obj, "use_lora", False) or (lora_r > 0)
        )
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.min_numel = max(1, int(min_numel or _cfg_get(cfg_obj, "min_numel", 16)))
        self.target_modules = target_modules

        if max_error is not None:
            self.max_error: Optional[float] = float(max_error)
        elif _cfg_get(cfg_obj, "max_error", None) is not None:
            self.max_error = float(_cfg_get(cfg_obj, "max_error"))
        else:
            self.max_error = None

        self.fallback_to_original = bool(
            fallback_to_original if fallback_to_original is not None
            else _cfg_get(cfg_obj, "fallback_to_original", True)
        )

        # Device
        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = (
                    next(model.parameters()).device
                    if list(model.parameters()) else torch.device("cpu")
                )
        else:
            self.target_device = (
                next(model.parameters()).device
                if list(model.parameters()) else torch.device("cpu")
            )

        self.visited_modules: Set[int] = set()
        self.compressed_layers: List[str] = []
        self.protected_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.failed_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_actual_sparsity: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if self.target_attention_only:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        is_gqa = (num_heads is not None and num_kv_heads is not None
                  and num_kv_heads < num_heads)

        name_l = name.lower()
        is_attn_qkv = any(p in name_l for p in (
            "q_proj", "k_proj", "v_proj", "c_attn", "query", "key", "value",
            "q_lin", "k_lin", "v_lin",
        ))
        return (is_gqa or self.skip_attention_qkv) and is_attn_qkv

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in (
            "classifier", "score", "lm_head", "pooler", "embed", "embedding",
            "cls_head", "head", "logits_processor",
        )):
            self.protected_layers.append(name)
            return False

        if self.target_attention_only:
            attn_keywords = (
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "c_attn", "c_proj", "query", "key", "value", "dense",
            )
            if not any(k in name_l for k in attn_keywords):
                self.skipped_layers.append(name)
                return False

        if self._is_gqa_asymmetric(name, module):
            self.skipped_layers.append(name)
            return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        raw_w = getattr(base_mod, "weight", None)
        if raw_w is None or not isinstance(raw_w, torch.Tensor):
            return False

        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
        is_linear_like = isinstance(base_mod, nn.Linear) or is_conv1d
        if not is_linear_like or isinstance(module, ProxSparsePrunedLoRALinear):
            return False

        if raw_w.numel() < self.min_numel:
            self.skipped_layers.append(name)
            return False

        if not should_compress_linear(self.model, name):
            self.protected_layers.append(name)
            return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    def _soft_threshold(self, tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        """prox_{a ||.||_1}(x) = sign(x) * max(|x| - a, 0)."""
        return torch.sign(tensor) * torch.clamp(torch.abs(tensor) - threshold, min=0.0)

    def _project_sparse(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard projection onto sparse constraint S. Returns (pruned, mask)."""
        if self.structured_24:
            pruned, mask = apply_24_structured_mask(
                tensor, saliency=torch.abs(tensor), dim=1
            )
            return pruned, (mask > 0)
        elif self.block_size != (1, 1):
            pruned, mask = apply_block_mask(
                tensor, block_size=self.block_size, sparsity=self.sparsity
            )
            return pruned, (mask > 0)
        else:
            flat = tensor.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = tensor.abs() >= thr
            return tensor * mask.float(), mask

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers.clear()
        self.visited_modules.clear()
        self.protected_layers.clear()
        self.skipped_layers.clear()
        self.fallback_layers.clear()
        self.failed_layers.clear()
        self.layer_errors.clear()
        self.layer_actual_sparsity.clear()

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            mod_id = id(module)
            if mod_id in self.visited_modules:
                self.skipped_layers.append(name)
                continue
            self.visited_modules.add(mod_id)

            try:
                base_mod = getattr(module, "base_layer", module)
                raw_w = base_mod.weight.data
                cls_name = base_mod.__class__.__name__
                is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
                device = raw_w.device
                dtype = raw_w.dtype

                # Always operate in (out, in) orientation.
                W_0 = raw_w.t().contiguous().float() if is_conv1d else raw_w.contiguous().float()

                # Early NaN/Inf guard on source.
                if not torch.isfinite(W_0).all():
                    logger.warning("ProxSparse: %s has non-finite weights; skipping.", name)
                    self.failed_layers.append(name)
                    continue

                # ------------------------------------------------
                # PS-FIX-01: Correct proximal gradient iteration.
                # ------------------------------------------------
                eta = self.learning_rate
                W = W_0.clone()
                mask = None
                for _ in range(self.steps):
                    grad_W = W - W_0                       # grad of 0.5*||W - W_0||^2
                    W_grad = W - eta * grad_W              # gradient descent step
                    W_prox = self._soft_threshold(W_grad, eta * self.lambda_prox)
                    W, mask = self._project_sparse(W_prox)

                # Final projection (mask from last iteration already applied).
                W_final = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

                w_norm = torch.norm(W_0, p="fro").clamp_min(1e-12).item()
                err = (torch.norm(W_0 - W_final, p="fro") / w_norm).item()
                self.layer_errors[name] = float(err)

                # Actual sparsity in the final weight.
                actual_zero = (W_final.abs() < 1e-12).float().mean().item()
                self.layer_actual_sparsity[name] = float(actual_zero)

                is_invalid = torch.isnan(W_final).any() or torch.isinf(W_final).any()
                if self.max_error is not None and err > self.max_error:
                    logger.warning(
                        "ProxSparse: reconstruction error %.4f on %s exceeded %.4f.",
                        err, name, self.max_error,
                    )
                    is_invalid = True

                if is_invalid and self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

                # ---- Commit ----
                if self.use_lora and self.lora_r > 0:
                    lora_layer = ProxSparsePrunedLoRALinear.from_linear(
                        linear=module,
                        pruned_weight=W_final.to(dtype=dtype, device=device),
                        mask=mask.to(device=device),
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    replace_module(self.model, name, lora_layer)
                else:
                    final_w = W_final.t().contiguous() if is_conv1d else W_final.contiguous()
                    base_mod.weight.data.copy_(final_w.to(dtype=dtype, device=device))

                self.compressed_layers.append(name)

            except Exception as e_comp:
                logger.warning("ProxSparse: failed on %s: %s", name, e_comp)
                self.failed_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # Forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy = torch.randint(0, 100, (1, 8), device=self.target_device)
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    self.model(dummy)
        except Exception as ex_sanity:
            logger.warning("ProxSparse forward sanity check failed: %s", ex_sanity)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )

        # PS-FIX-08: effective_parameter_ratio separate from compression_ratio.
        effective_ratio = 1.0
        if self.compressed_layers:
            ratios = []
            for n in self.compressed_layers:
                actual_sp = self.layer_actual_sparsity.get(n, 0.0)
                # Non-zeros in the pruned base + LoRA params (if any)
                ratio = (1.0 - actual_sp)
                if self.use_lora and self.lora_r > 0:
                    # Very rough estimate; LoRA adds lora_r*(in+out) params.
                    ratio += 0.02  # approximate placeholder; real accounting per layer
                ratios.append(ratio)
            effective_ratio = sum(ratios) / len(ratios)

        stats = self.get_compression_stats()
        stats.update({
            "effective_parameter_ratio": round(effective_ratio, 4),
            "pruned_layers": self.compressed_layers,
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "protected_layers": self.protected_layers,
            "protected_layer_count": len(self.protected_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "failed_layers": self.failed_layers,
            "failed_layer_count": len(self.failed_layers),
            "layer_errors": self.layer_errors,
            "layer_actual_sparsity": self.layer_actual_sparsity,
            "avg_reconstruction_error": avg_err,
            "forward_check_success": forward_check_success,
            "sparsity_target": self.sparsity,
            "lambda_prox": self.lambda_prox,
            "learning_rate": self.learning_rate,
            "structured_24": self.structured_24,
            "steps": self.steps,
            "skip_attention_qkv": self.skip_attention_qkv,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "method": "proxsparse_regularized",
        })
        self.stats = stats
        return self.model, stats


# Aliases
ProxSparseCompressor = ProxSparsePruner


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("proxsparse_factory")
@COMPRESSOR_REGISTRY.register("proxsparse_pruner_factory")
@COMPRESSOR_REGISTRY.register("proxsparse_compressor_factory")
def proxsparse_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "proxsparse_sparsity",
                 _cfg_get(comp_config, "sparsity", 0.5)),
    )
    lambda_prox = kwargs.pop(
        "lambda_prox",
        _cfg_get(comp_config, "proxsparse_lambda",
                 _cfg_get(comp_config, "lambda_prox", 0.01)),
    )
    steps = kwargs.pop(
        "steps",
        _cfg_get(comp_config, "proxsparse_steps",
                 _cfg_get(comp_config, "steps", 5)),
    )
    learning_rate = kwargs.pop(
        "learning_rate",
        _cfg_get(comp_config, "proxsparse_lr",
                 _cfg_get(comp_config, "learning_rate", 0.5)),
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(comp_config, "proxsparse_structured_24",
                 _cfg_get(comp_config, "structured_24", False)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv",
        _cfg_get(comp_config, "skip_attention_qkv", True),
    )
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", None))
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "proxsparse_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )

    compressor = ProxSparsePruner(
        model,
        sparsity=sparsity,
        lambda_prox=lambda_prox,
        steps=steps,
        learning_rate=learning_rate,
        structured_24=structured_24,
        target_attention_only=target_attention_only,
        skip_attention_qkv=skip_attention_qkv,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "ProxSparsePrunedLoRALinear",
    "ProxSparsePruner",
    "ProxSparseCompressor",
    "proxsparse_factory",
]