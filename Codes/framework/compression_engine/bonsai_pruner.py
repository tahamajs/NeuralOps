"""
Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradient Descent Recovery
& [CLS] Routing Protection. Registered in COMPRESSOR_REGISTRY as 'bonsai' and 'bonsai_pruner'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
    compute_taylor_first_order_saliency,
    get_calibration_tensors,
    recalibrate_layernorms,
    adaptive_max_error,
)
from framework.utils.compat import should_compress_linear


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


logger = logging.getLogger(__name__)


# ===========================================================================
# BonsaiPrunedLoRALinear
# ===========================================================================

class BonsaiPrunedLoRALinear(nn.Module):
    """
    Linear with sparse pruned base weight (frozen) + optional trainable LoRA adapter.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features                # Conv1D compatibility
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / max(1, self.lora_r)) if self.lora_r > 0 else 1.0

        weight_dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=weight_dtype, device=device)
        )
        # BN-FIX-02/03: non-persistent mask in the same dtype as weight.
        self.register_buffer(
            "mask",
            torch.ones(out_features, in_features, dtype=weight_dtype, device=device),
            persistent=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=weight_dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_r, in_features, dtype=weight_dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.empty(out_features, self.lora_r, dtype=weight_dtype, device=device)
            )
            with torch.no_grad():
                tmp_A = self.lora_A.data.float()
                nn.init.kaiming_uniform_(tmp_A, a=math.sqrt(5))
                self.lora_A.data.copy_(tmp_A.to(weight_dtype))
                self.lora_B.data.zero_()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    @property
    def effective_weight(self) -> torch.Tensor:
        W_eff = self.weight * self.mask
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            W_eff = W_eff + (torch.matmul(self.lora_B, self.lora_A) * self.scaling)
        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        W_sparse = self.weight * self.mask
        out = F.linear(x_flat, W_sparse, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out.view(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear | Any,
        pruned_weight: torch.Tensor,
        mask: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ) -> BonsaiPrunedLoRALinear:
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
        with torch.no_grad():
            layer.weight.data.copy_(pruned_weight)
            layer.mask.copy_(mask.to(pruned_weight.dtype))
            if has_bias:
                layer.bias.data.copy_(linear.bias.data)
        return layer


# ===========================================================================
# BonsaiPruner
# ===========================================================================

@COMPRESSOR_REGISTRY.register("bonsai")
@COMPRESSOR_REGISTRY.register("bonsai_pruner")
class BonsaiPruner(BaseCompressor):
    """
    Bonsai Pruner: progressive multi-step iterative pruning with cubic schedule,
    inter-step gradient descent recovery, and [CLS]/GQA protection.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        steps: int = 8,
        recovery_steps: int = 3,
        recovery_lr: float = 1e-3,
        structured_24: bool = False,
        block_size: tuple[int, int] = (4, 4),
        protect_cls: bool = True,
        skip_attention_qkv: bool = True,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        adaptive_sparsity: bool = False,
        min_numel: int = 16,
        max_error: float | None = None,
        fallback_to_original: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Any | None = None,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        cfg: Any | None = None,
        max_recovery_steps: int = 32,       # BN-FIX-05
        max_step_size: float = 0.01,        # BN-FIX-06
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="bonsai", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config")

        self.sparsity = max(0.0, min(float(sparsity or getattr(cfg_obj, "sparsity", 0.5)), 0.95))
        self.steps = max(1, int(steps or getattr(cfg_obj, "steps", 8)))
        raw_recovery = int(recovery_steps if recovery_steps is not None else getattr(cfg_obj, "recovery_steps", 3))
        self.recovery_steps = max(0, min(raw_recovery, max_recovery_steps))
        self.recovery_lr = float(recovery_lr or getattr(cfg_obj, "recovery_lr", 1e-3))
        self.structured_24 = bool(structured_24 if structured_24 is not None else getattr(cfg_obj, "structured_24", False))
        self.block_size = block_size or getattr(cfg_obj, "block_size", (4, 4))
        self.protect_cls = bool(protect_cls if protect_cls is not None else getattr(cfg_obj, "protect_cls", True))
        self.target_attention_only = bool(target_attention_only)
        self.skip_attention_qkv = bool(
            skip_attention_qkv if skip_attention_qkv is not None
            else getattr(cfg_obj, "skip_attention_qkv", False)
        )
        self.use_lora = bool(use_lora or getattr(cfg_obj, "use_lora", False) or (lora_r > 0))
        self.lora_r = int(lora_r or getattr(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or getattr(cfg_obj, "lora_alpha", 1.0))
        self.adaptive_sparsity = bool(adaptive_sparsity or getattr(cfg_obj, "adaptive_sparsity", False))
        self.min_numel = max(1, int(min_numel or getattr(cfg_obj, "min_numel", 16)))
        self.max_recovery_steps = max(1, int(max_recovery_steps))
        self.max_step_size = float(max_step_size)

        if max_error is not None:
            self.max_error: float | None = float(max_error)
        elif hasattr(cfg_obj, "max_error") and cfg_obj.max_error is not None:
            self.max_error = float(cfg_obj.max_error)
        else:
            self.max_error = adaptive_max_error(int((1.0 - self.sparsity) * 768), 768, 768, "distilbert")

        self.fallback_to_original = bool(
            fallback_to_original if fallback_to_original is not None
            else getattr(cfg_obj, "fallback_to_original", True)
        )
        self.recalibrate_ln = bool(
            recalibrate_ln if recalibrate_ln is not None
            else getattr(cfg_obj, "recalibrate_ln", True)
        )

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts

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

        self.pruned_layers: list[str] = []
        self.protected_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_sparsities: dict[str, float] = {}
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # BN-FIX-01: independent filters.
    # ------------------------------------------------------------------
    _ATTN_TOKENS = (
        "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
        "out_proj", "q_lin", "k_lin", "v_lin", "out_lin",
        "query", "key", "value", "c_attn", "c_proj",
    )

    _PROTECT_TOKENS = (
        "router", "gate", "cls", "classifier", "head",
        "embed", "score", "lm_head", "pooler",
    )

    _QKV_TOKENS = (
        "query", "key", "value",
        "q_lin", "k_lin", "v_lin",
        "q_proj", "k_proj", "v_proj",
        "c_attn",
    )

    def _is_protected_layer(self, name: str, module: nn.Module | None = None) -> bool:
        """
        A layer is protected if ANY of the following is true:
          (a) protect_cls=True AND name matches protect tokens.
          (b) target_attention_only=True AND name is NOT an attention projection.
          (c) skip_attention_qkv=True AND name is a QKV projection.
          (d) should_compress_linear returns False.
        The filters are independent (BN-FIX-01).
        """
        name_lower = name.lower()

        # (a) explicit protect tokens.
        if self.protect_cls and any(kw in name_lower for kw in self._PROTECT_TOKENS):
            return True

        # (b) attention-only mode.
        if self.target_attention_only and not any(k in name_lower for k in self._ATTN_TOKENS):
            return True

        # (c) QKV protection.
        if self.skip_attention_qkv:
            last_part = name_lower.split(".")[-1]
            if any(k in last_part for k in self._QKV_TOKENS):
                return True
            if module is not None and hasattr(module, "weight") and module.weight.ndim == 2:
                if ("attn" in name_lower or "attention" in name_lower):
                    w_shape = module.weight.shape
                    if w_shape[0] != w_shape[1]:  # GQA asymmetry
                        return True

        # (d) framework guard.
        try:
            if not should_compress_linear(self.model, name):
                return True
        except Exception:
            pass

        return False

    # ------------------------------------------------------------------
    def _collect_activations(self) -> dict[str, torch.Tensor]:
        self.model.eval()
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inp, outp):
                if inp and isinstance(inp[0], torch.Tensor):
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(inp[0].detach().cpu())
            return hook

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__
            is_linear_like = (
                isinstance(module, nn.Linear)
                or "conv1d" in cls_name.lower()
                or hasattr(module, "nf")
            )
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

                if not forward_success and isinstance(calib_inputs, dict):
                    try:
                        subset = {k: calib_inputs[k] for k in ("input_ids", "attention_mask") if k in calib_inputs}
                        if subset:
                            self.model(**subset)
                            forward_success = True
                    except Exception as e2:
                        logger.debug("Bonsai calibration strategy 2 failed: %s", e2)

                if not forward_success and isinstance(calib_inputs, dict) and "input_ids" in calib_inputs:
                    try:
                        self.model(calib_inputs["input_ids"])
                        forward_success = True
                    except Exception as e3:
                        logger.debug("Bonsai calibration strategy 3 failed: %s", e3)

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

        aggregated: dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(
                    [a.reshape(-1, a.shape[-1]) for a in act_list], dim=0
                )
        return aggregated

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        try:
            activations_map = self._collect_activations()
        except Exception as e:
            logger.warning("BonsaiPruner: activation collection failed (%s); proceeding with unweighted pruning.", e)
            activations_map = {}

        self.pruned_layers = []
        self.protected_layers = []
        self.skipped_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_linear_like = (
                isinstance(module, nn.Linear)
                or "conv1d" in cls_name.lower()
                or hasattr(module, "nf")
            )
            if not is_linear_like or isinstance(module, BonsaiPrunedLoRALinear):
                continue

            if self._is_protected_layer(name, module):
                for p in module.parameters():
                    p.requires_grad = False
                self.protected_layers.append(name)
                continue

            target_mod = getattr(module, "base_layer", module)
            raw_w = target_mod.weight.data
            dtype = raw_w.dtype
            device = raw_w.device
            is_conv1d = "conv1d" in cls_name.lower() or hasattr(module, "nf")

            W0 = raw_w.t().float().contiguous() if is_conv1d else raw_w.float().contiguous()
            out_f, in_f = W0.shape

            if (out_f * in_f) < self.min_numel:
                for p in module.parameters():
                    p.requires_grad = False
                self.skipped_layers.append(name)
                continue

            W = W0.clone()
            act = activations_map.get(name)
            X = None
            if act is not None and act.numel() > 0:
                X = act.to(device).float()
                if X.shape[-1] != in_f:
                    X = None

            # BN-FIX-10: bounded adaptive sparsity.
            if self.adaptive_sparsity and X is not None:
                act_norm = torch.norm(X, p="fro").item() / max(1, X.shape[0])
                scale_factor = 1.0 / (1.0 + math.log1p(max(0.0, act_norm)))
                target_sp = max(0.05, min(0.95, self.sparsity * (0.7 + 0.6 * scale_factor)))
            else:
                target_sp = self.sparsity

            final_mask = torch.ones_like(W)

            # Iterative pruning loop.
            for s in range(self.steps):
                t_ratio = (s + 1.0) / self.steps
                current_sp = target_sp * (1.0 - (1.0 - t_ratio) ** 3)
                saliency = compute_taylor_first_order_saliency(W, activations=X)

                if self.structured_24:
                    W_pruned, final_mask = apply_24_structured_mask(W, saliency=saliency, dim=1)
                elif self.block_size != (1, 1):
                    W_pruned, final_mask = apply_block_mask(
                        W, block_size=self.block_size,
                        sparsity=current_sp, saliency=saliency,
                    )
                else:
                    flat = saliency.reshape(-1)
                    k = max(1, int(flat.numel() * (1.0 - current_sp)))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    final_mask = (saliency >= thr).float()
                    W_pruned = W * final_mask

                # BN-FIX-04: explicit clone.
                W = W_pruned.clone()

                # Gradient descent recovery between steps.
                if self.recovery_steps > 0 and X is not None:
                    cov_X = torch.matmul(X.t(), X) / max(X.shape[0], 1)
                    W0_cov = torch.matmul(W0, cov_X)

                    # BN-FIX-06: clamp step size.
                    raw_ss = self.recovery_lr / (torch.norm(cov_X).item() + 1e-6)
                    step_size = min(raw_ss, self.max_step_size)
                    for _ in range(self.recovery_steps):
                        curr_cov = torch.matmul(W, cov_X)
                        grad = (curr_cov - W0_cov) * final_mask
                        W = W - step_size * grad
                        W = W * final_mask

            norm_orig = torch.norm(W0, p="fro").clamp_min(1e-8)
            rel_err = (torch.norm(W0 - W, p="fro") / norm_orig).item()
            self.layer_errors[name] = float(rel_err)
            self.layer_sparsities[name] = float(target_sp)

            is_invalid = (not torch.isfinite(W).all()) or (rel_err > (self.max_error or float("inf")))
            if is_invalid:
                logger.warning(
                    "BonsaiPruner: %s error %.4f (threshold %.4f).",
                    name, rel_err,
                    self.max_error if self.max_error is not None else float("inf"),
                )
                if self.fallback_to_original:
                    for p in module.parameters():
                        p.requires_grad = False
                    self.fallback_layers.append(name)
                    continue

            # Commit layer.
            if self.use_lora and self.lora_r > 0:
                pruned_lora_layer = BonsaiPrunedLoRALinear.from_linear(
                    linear=target_mod,
                    pruned_weight=W.to(dtype=dtype, device=device),
                    mask=final_mask.to(device=device),
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    freeze_base=True,
                )
                # BN-FIX-07: copy LoRA data (no shared reference).
                if hasattr(module, "lora_A") and module.lora_A is not None \
                        and pruned_lora_layer.lora_A is not None:
                    try:
                        src_A = getattr(module.lora_A, "weight", module.lora_A)
                        pruned_lora_layer.lora_A.data.copy_(src_A.data)
                    except Exception as e:
                        logger.debug("Bonsai LoRA A transfer failed: %s", e)
                if hasattr(module, "lora_B") and module.lora_B is not None \
                        and pruned_lora_layer.lora_B is not None:
                    try:
                        src_B = getattr(module.lora_B, "weight", module.lora_B)
                        pruned_lora_layer.lora_B.data.copy_(src_B.data)
                    except Exception as e:
                        logger.debug("Bonsai LoRA B transfer failed: %s", e)

                if hasattr(module, "base_layer"):
                    module.base_layer = pruned_lora_layer
                    pruned_lora_layer = module
                replace_module(self.model, name, pruned_lora_layer)
            else:
                # BN-FIX-08: no_grad for weight mutation.
                with torch.no_grad():
                    if is_conv1d:
                        target_mod.weight.data.copy_(W.t().contiguous().to(dtype=dtype, device=device))
                    else:
                        target_mod.weight.data.copy_(W.contiguous().to(dtype=dtype, device=device))

            self.pruned_layers.append(name)

        # LayerNorm recalibration.
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

        # BN-FIX-09: forward sanity check with attention_mask.
        sanity_check_passed = True
        try:
            with torch.no_grad():
                dummy = torch.randint(0, 100, (1, 8), device=self.target_device, dtype=torch.long)
                attn = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn)
                except Exception:
                    self.model(dummy)
        except Exception as ex:
            logger.debug("BonsaiPruner post-compression forward sanity check warning: %s", ex)
            sanity_check_passed = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
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
        logger.info(
            "Compressor bonsai finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.pruned_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error if self.max_error is not None else 1.0,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("bonsai")
@COMPRESSOR_REGISTRY.register("bonsai_pruner")
@COMPRESSOR_REGISTRY.register("bonsai_factory")
@COMPRESSOR_REGISTRY.register("bonsai_pruner_factory")
def bonsai_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Factory for Bonsai pruner."""
    return_both = kwargs.pop("return_both", False)

    def _get(key: str, default: Any) -> Any:
        if comp_config is None:
            return kwargs.pop(key, default)
        return kwargs.pop(key, getattr(comp_config, key, getattr(comp_config, f"bonsai_{key}", default)))

    sparsity = _get("sparsity", 0.5)
    steps = _get("steps", 8)
    recovery_steps = _get("recovery_steps", 3)
    recovery_lr = _get("recovery_lr", 1e-3)
    structured_24 = _get("structured_24", False)
    protect_cls = _get("protect_cls", True)
    recalibrate_ln = _get("recalibrate_ln", True)
    skip_attention_qkv = _get("skip_attention_qkv", False)
    use_lora = _get("use_lora", False)
    lora_r = _get("lora_r", 0)
    lora_alpha = _get("lora_alpha", 1.0)
    adaptive_sparsity = _get("adaptive_sparsity", False)
    max_error = _get("max_error", None)
    fallback = _get("fallback_to_original", True)

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
        recalibrate_ln=recalibrate_ln,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_stats or return_tuple or return_both:
        return compressed_model, stats
    return compressed_model


bonsai_prune_model = bonsai_factory

__all__ = [
    "BonsaiPrunedLoRALinear",
    "BonsaiPruner",
    "bonsai_factory",
    "bonsai_prune_model",
]
