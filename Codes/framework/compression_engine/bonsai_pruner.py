"""
Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradient Descent Recovery & [CLS] Routing Protection.
Registered in COMPRESSOR_REGISTRY as 'bonsai' and 'bonsai_pruner'.

Production-hardened with:
- Multi-architecture support: DistilBERT, RoBERTa, GPT-2 (Conv1D), Qwen3.5 (GQA)
- Robust activation collection & calibration forward handling (never crashes on malformed inputs)
- Safe LayerNorm recalibration with try/except guards
- GQA-aware handling (configurable skip_attention_qkv to preserve KV-cache alignment)
- Trainable LoRA adapter integration post-pruning (BonsaiPrunedLoRALinear)
- Guard against tiny layers (< 16 parameters)
- Clamped step size scaling in recovery loop
- Per-layer adaptive sparsity option based on activation sensitivity
- Post-compression forward sanity check
- Robust device management
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
    compute_taylor_first_order_saliency,
    get_calibration_tensors,
    recalibrate_layernorms,
)
from framework.utils.compat import should_compress_linear


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


logger = logging.getLogger(__name__)


class BonsaiPrunedLoRALinear(nn.Module):
    """
    Linear layer with sparse pruned base weights and optional trainable LoRA adapters.
    Base weights are frozen with persistent sparse mask; LoRA adapters (lora_A, lora_B)
    are trainable with initial zero delta contribution for efficient post-pruning recovery.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / max(1, self.lora_r)) if self.lora_r > 0 else 1.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.register_buffer("mask", torch.ones(out_features, in_features, dtype=torch.float32, device=device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_r, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_r, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    @property
    def effective_weight(self) -> torch.Tensor:
        """Effective reconstructed weight matrix for inspection."""
        W_eff = self.weight * self.mask.to(dtype=self.weight.dtype)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            W_eff = W_eff + (torch.matmul(self.lora_B, self.lora_A) * self.scaling)
        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        W_sparse = self.weight * self.mask.to(dtype=self.weight.dtype)
        out = F.linear(x_flat, W_sparse, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out.view(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        pruned_weight: torch.Tensor,
        mask: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ) -> "BonsaiPrunedLoRALinear":
        out_features, in_features = pruned_weight.shape
        has_bias = getattr(linear, "bias", None) is not None and linear.bias is not None

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base,
            dtype=pruned_weight.dtype,
            device=pruned_weight.device,
        )
        layer.weight.data.copy_(pruned_weight)
        layer.mask.copy_(mask.float())
        if has_bias:
            layer.bias.data.copy_(linear.bias.data)
        return layer


@COMPRESSOR_REGISTRY.register("bonsai")
@COMPRESSOR_REGISTRY.register("bonsai_pruner")
class BonsaiPruner(BaseCompressor):
    """
    Bonsai Pruner: Progressive multi-step iterative pruning with cubic schedule,
    inter-step gradient descent recovery, GQA-aware protection, and [CLS] routing protection.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        steps: int = 8,
        recovery_steps: int = 3,
        recovery_lr: float = 1e-3,
        structured_24: bool = False,
        block_size: Tuple[int, int] = (4, 4),
        protect_cls: bool = True,
        skip_attention_qkv: bool = True,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        adaptive_sparsity: bool = False,
        min_numel: int = 16,
        max_error: Optional[float] = None,
        fallback_to_original: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="bonsai", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config", None)

        self.sparsity = max(0.0, min(float(sparsity or getattr(cfg_obj, "sparsity", 0.5)), 0.95))
        self.steps = max(1, int(steps or getattr(cfg_obj, "steps", 8)))
        self.recovery_steps = max(0, int(recovery_steps if recovery_steps is not None else getattr(cfg_obj, "recovery_steps", 3)))
        self.recovery_lr = float(recovery_lr or getattr(cfg_obj, "recovery_lr", 1e-3))
        self.structured_24 = bool(structured_24 if structured_24 is not None else getattr(cfg_obj, "structured_24", False))
        self.block_size = block_size or getattr(cfg_obj, "block_size", (4, 4))
        self.protect_cls = bool(protect_cls if protect_cls is not None else getattr(cfg_obj, "protect_cls", True))
        self.target_attention_only = target_attention_only
        self.skip_attention_qkv = bool(
            skip_attention_qkv if skip_attention_qkv is not None else getattr(cfg_obj, "skip_attention_qkv", True)
        )
        self.use_lora = bool(use_lora or getattr(cfg_obj, "use_lora", False) or (lora_r > 0))
        self.lora_r = int(lora_r or getattr(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or getattr(cfg_obj, "lora_alpha", 1.0))
        self.adaptive_sparsity = bool(adaptive_sparsity or getattr(cfg_obj, "adaptive_sparsity", False))
        self.min_numel = max(1, int(min_numel or getattr(cfg_obj, "min_numel", 16)))
        if max_error is not None:
            self.max_error: Optional[float] = float(max_error)
        elif hasattr(cfg_obj, "max_error") and getattr(cfg_obj, "max_error") is not None:
            self.max_error = float(getattr(cfg_obj, "max_error"))
        else:
            self.max_error = None
        self.fallback_to_original = bool(
            fallback_to_original if fallback_to_original is not None else getattr(cfg_obj, "fallback_to_original", True)
        )
        self.recalibrate_ln = bool(recalibrate_ln if recalibrate_ln is not None else getattr(cfg_obj, "recalibrate_ln", True))

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts

        # Robust device derivation
        try:
            default_dev = next(model.parameters()).device
        except (StopIteration, AttributeError):
            default_dev = torch.device("cpu")

        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = default_dev
        else:
            self.target_device = default_dev

        self.pruned_layers: List[str] = []
        self.protected_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_sparsities: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_protected_layer(self, name: str, module: Optional[nn.Module] = None) -> bool:
        """Determines if a layer is a protected [CLS], router, GQA projection, or head module."""
        if not self.protect_cls:
            if self.target_attention_only and not any(k in name.lower() for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "c_attn", "c_proj"]):
                return True
            return False

        name_lower = name.lower()
        protected_keywords = [
            "router", "gate", "cls", "classifier", "head", "embed", "score", "lm_head", "pooler"
        ]
        if any(kw in name_lower for kw in protected_keywords):
            return True

        if not should_compress_linear(self.model, name):
            return True

        if self.skip_attention_qkv:
            attn_keywords = [
                "query", "key", "value", "q_lin", "k_lin", "v_lin", "q_proj", "k_proj", "v_proj", "c_attn"
            ]
            last_part = name_lower.split(".")[-1]
            if any(k in last_part for k in attn_keywords):
                return True
            # Detect GQA projection asymmetry
            if module is not None and hasattr(module, "weight") and module.weight.ndim == 2:
                if "attn" in name_lower or "attention" in name_lower:
                    w_shape = module.weight.shape
                    if w_shape[0] != w_shape[1]:  # Asymmetric Q vs KV in GQA
                        return True

        return False

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Capture activations across linear layers with robust multi-strategy execution."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(x.cpu())
            return hook

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower() or hasattr(module, "nf")
            if is_linear_like:
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

            with torch.no_grad():
                forward_success = False
                # Strategy 1: Standard model forward invocation
                try:
                    if isinstance(calib_inputs, dict):
                        self.model(**calib_inputs)
                    elif isinstance(calib_inputs, torch.Tensor):
                        self.model(calib_inputs)
                    elif isinstance(calib_inputs, (list, tuple)):
                        self.model(*calib_inputs)
                    forward_success = True
                except Exception as e1:
                    logger.debug("Bonsai calibration strategy 1 failed: %s", e1)

                # Strategy 2: Common text model keys (input_ids, attention_mask)
                if not forward_success and isinstance(calib_inputs, dict):
                    try:
                        subset = {}
                        for k in ("input_ids", "attention_mask"):
                            if k in calib_inputs:
                                subset[k] = calib_inputs[k]
                        if subset:
                            self.model(**subset)
                            forward_success = True
                    except Exception as e2:
                        logger.debug("Bonsai calibration strategy 2 failed: %s", e2)

                # Strategy 3: Direct tensor passing
                if not forward_success and isinstance(calib_inputs, dict) and "input_ids" in calib_inputs:
                    try:
                        self.model(calib_inputs["input_ids"])
                        forward_success = True
                    except Exception as e3:
                        logger.debug("Bonsai calibration strategy 3 failed: %s", e3)

                # Strategy 4: Synthetic dummy fallback
                if not forward_success:
                    try:
                        dummy = torch.randint(0, 100, (1, 16), device=self.target_device)
                        self.model(dummy)
                    except Exception as e4:
                        logger.debug("Bonsai synthetic calibration failed: %s", e4)
        except Exception as e:
            logger.warning("Bonsai calibration exception: %s", e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes progressive iterative pruning with inter-step gradient descent recovery."""
        try:
            activations_map = self._collect_activations()
        except Exception as e:
            logger.warning("BonsaiPruner: Activation collection failed (%s); proceeding with unweighted pruning.", e)
            activations_map = {}

        self.pruned_layers = []
        self.protected_layers = []
        self.skipped_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}
        pruned_layers_count = 0
        protected_layers_count = 0

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower() or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, BonsaiPrunedLoRALinear):
                continue

            if self._is_protected_layer(name, module):
                protected_layers_count += 1
                for p in module.parameters(): p.requires_grad = False
                self.protected_layers.append(name)
                continue

            target_mod = getattr(module, 'base_layer', module)
            raw_w = target_mod.weight.data
            dtype = raw_w.dtype
            device = raw_w.device
            is_conv1d = "conv1d" in cls_name.lower() or hasattr(module, "nf")

            # Conv1D orientation: [in_features, out_features] -> [out_features, in_features]
            W0 = raw_w.t().float().contiguous() if is_conv1d else raw_w.float().contiguous()
            out_f, in_f = W0.shape

            # Guard against tiny layers
            if (out_f * in_f) < self.min_numel:
                for p in module.parameters(): p.requires_grad = False
                self.skipped_layers.append(name)
                continue

            W = W0.clone()
            act = activations_map.get(name, None)
            X = None
            if act is not None and act.numel() > 0:
                X = act.to(device).float()
                if X.shape[-1] != in_f:
                    X = None

            # Per-layer adaptive sparsity calculation
            if self.adaptive_sparsity and X is not None:
                act_norm = torch.norm(X, p="fro").item() / max(1, X.shape[0])
                scale_factor = 1.0 / (1.0 + math.log1p(max(0.0, act_norm)))
                target_sp = max(0.05, min(0.95, self.sparsity * (0.7 + 0.6 * scale_factor)))
            else:
                target_sp = self.sparsity

            final_mask = torch.ones_like(W)

            # Iterative Pruning Loop
            for s in range(self.steps):
                t_ratio = (s + 1.0) / self.steps
                current_sp = target_sp * (1.0 - (1.0 - t_ratio) ** 3)
                saliency = compute_taylor_first_order_saliency(W, activations=X)

                # 1. Compute Mask for step s
                if self.structured_24:
                    W_pruned, final_mask = apply_24_structured_mask(W, saliency=saliency, dim=1)
                elif self.block_size != (1, 1):
                    W_pruned, final_mask = apply_block_mask(
                        W, block_size=self.block_size, sparsity=current_sp, saliency=saliency
                    )
                else:
                    flat = saliency.reshape(-1)
                    k = max(1, int(flat.numel() * (1.0 - current_sp)))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    final_mask = (saliency >= thr).float()
                    W_pruned = W * final_mask

                W = W_pruned

                # 2. Gradient Descent Recovery between pruning steps
                if self.recovery_steps > 0 and X is not None:
                    cov_X = torch.matmul(X.T, X) / max(X.shape[0], 1)
                    W0_cov = torch.matmul(W0, cov_X)

                    # Clamped step size scaling
                    step_size = min(self.recovery_lr / (torch.norm(cov_X).item() + 1e-6), 1.0)
                    for _ in range(self.recovery_steps):
                        curr_cov = torch.matmul(W, cov_X)
                        grad = (curr_cov - W0_cov) * final_mask
                        W = W - step_size * grad
                        W = W * final_mask  # enforce sparse topology

            # Numerical sanity and reconstruction validation
            norm_orig = torch.norm(W0, p="fro").clamp_min(1e-8)
            rel_err = (torch.norm(W0 - W, p="fro") / norm_orig).item()
            self.layer_errors[name] = float(rel_err)
            self.layer_sparsities[name] = float(target_sp)

            is_invalid = torch.isnan(W).any() or torch.isinf(W).any()
            if self.max_error is not None and rel_err > self.max_error:
                is_invalid = True
                logger.warning(
                    "BonsaiPruner: %s error (%.4f) exceeded threshold (%.4f).",
                    name, rel_err, self.max_error
                )

            if is_invalid and self.fallback_to_original:
                for p in module.parameters(): p.requires_grad = False
                self.fallback_layers.append(name)
                continue

            # Commit pruned layer with optional LoRA integration
            if self.use_lora and self.lora_r > 0:
                pruned_lora_layer = BonsaiPrunedLoRALinear.from_linear(
                    linear=target_mod,
                    pruned_weight=W.to(dtype=dtype, device=device),
                    mask=final_mask.to(device=device),
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    freeze_base=True,
                )
                if hasattr(module, 'base_layer'):
                    if hasattr(module, 'lora_A'):
                        pruned_lora_layer.lora_A = module.lora_A
                    if hasattr(module, 'lora_B'):
                        pruned_lora_layer.lora_B = module.lora_B
                    module.base_layer = pruned_lora_layer
                    pruned_lora_layer = module
                replace_module(self.model, name, pruned_lora_layer)
            else:
                if is_conv1d:
                    target_mod.weight.data.copy_(W.t().contiguous().to(dtype=dtype, device=device))
                else:
                    target_mod.weight.data.copy_(W.contiguous().to(dtype=dtype, device=device))

            self.pruned_layers.append(name)
            pruned_layers_count += 1

        # LayerNorm Recalibration with safety guard
        if self.recalibrate_ln:
            try:
                recalibrate_layernorms(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
            except Exception as e:
                logger.warning("BonsaiPruner: LayerNorm recalibration skipped: %s", e)

        # Post-compression forward sanity check
        sanity_check_passed = True
        try:
            with torch.no_grad():
                dummy_input = torch.randint(0, 100, (1, 8), device=self.target_device)
                self.model(dummy_input)
        except Exception as ex:
            logger.debug("BonsaiPruner post-compression forward sanity check warning: %s", ex)
            sanity_check_passed = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": self.pruned_layers,
            "pruned_layer_count": len(self.pruned_layers),
            "protected_layers": self.protected_layers,
            "protected_layer_count": len(self.protected_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "layer_sparsities": self.layer_sparsities,
            "avg_reconstruction_error": avg_err,
            "sanity_check_passed": sanity_check_passed,
            "steps": self.steps,
            "recovery_steps": self.recovery_steps,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "protect_cls": self.protect_cls,
            "skip_attention_qkv": self.skip_attention_qkv,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "adaptive_sparsity": self.adaptive_sparsity,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "method": "bonsai_iterative_pruning_with_gd_recovery",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("bonsai_factory")
@COMPRESSOR_REGISTRY.register("bonsai_pruner_factory")
def bonsai_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """
    Factory function for Bonsai pruner.
    Returns (compressed_model, stats) if return_stats or return_tuple is True, else compressed_model.
    """
    sparsity = kwargs.pop("sparsity", getattr(comp_config, "bonsai_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else 0.5)
    steps = kwargs.pop("steps", getattr(comp_config, "bonsai_steps", getattr(comp_config, "steps", 8)) if comp_config else 8)
    recovery_steps = kwargs.pop("recovery_steps", getattr(comp_config, "bonsai_recovery_steps", getattr(comp_config, "recovery_steps", 3)) if comp_config else 3)
    recovery_lr = kwargs.pop("recovery_lr", getattr(comp_config, "bonsai_recovery_lr", getattr(comp_config, "recovery_lr", 1e-3)) if comp_config else 1e-3)
    structured_24 = kwargs.pop("structured_24", getattr(comp_config, "bonsai_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else False)
    protect_cls = kwargs.pop("protect_cls", getattr(comp_config, "bonsai_protect_cls", getattr(comp_config, "protect_cls", True)) if comp_config else True)
    recalibrate_ln = kwargs.pop("recalibrate_ln", getattr(comp_config, "bonsai_recalibrate_ln", getattr(comp_config, "recalibrate_ln", True)) if comp_config else True)
    skip_attention_qkv = kwargs.pop("skip_attention_qkv", getattr(comp_config, "skip_attention_qkv", True) if comp_config else True)
    use_lora = kwargs.pop("use_lora", getattr(comp_config, "use_lora", False) if comp_config else False)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    adaptive_sparsity = kwargs.pop("adaptive_sparsity", getattr(comp_config, "adaptive_sparsity", False) if comp_config else False)
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", None) if comp_config else None)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config else True)

    compressor = BonsaiPruner(
        model,
        sparsity=sparsity,
        steps=steps,
        recovery_steps=recovery_steps,
        recovery_lr=recovery_lr,
        structured_24=structured_24,
        protect_cls=protect_cls,
        skip_attention_qkv=skip_attention_qkv,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        adaptive_sparsity=adaptive_sparsity,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


bonsai_prune_model = bonsai_factory

__all__ = [
    "BonsaiPrunedLoRALinear",
    "BonsaiPruner",
    "bonsai_factory",
    "bonsai_prune_model",
]
