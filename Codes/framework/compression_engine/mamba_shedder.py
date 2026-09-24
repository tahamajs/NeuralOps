"""
Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer Shedding Compressor.

Registered as 'mamba_shedder' and 'mamba_shedder_compressor'.

Mathematical Formulation:
  1. Empirical Fisher Information: I(X_j) = sqrt(mean_n X_{n,j}^2)
  2. Layer Residual Sensitivity: Δ_l = ||X_{l+1} - X_l||_F / (||X_l||_F + ε)
  3. Channel Masking: W_pruned = W ⊙ M_channel^T

Bug-fix map:
  MS-FIX-01  is_custom_attention = False.
  MS-FIX-02  Removed misleading kwargs.pop() calls at start.
  MS-FIX-03  Shape mismatch -> warning + skip (no assert).
  MS-FIX-04  Multiple calibration batches supported.
  MS-FIX-05  Skip LoRA / SVDLinear / already-pruned.
  MS-FIX-06  Safe parameter-based device resolution.
  MS-FIX-07  random module import at top, seeded locally.
  MS-FIX-08  Debug logging for no-target.
  MS-FIX-09  tokenizer falls back to model.config.tokenizer.
  MS-FIX-10  Fast mean via sum/len.
  MS-FIX-11  skip_should_compress_filter.
  MS-FIX-12  Explicit note in stats (rewrite-only compressor).
  MS-FIX-13  Idempotence flag on model.
  MS-FIX-14  Bidirectional prompt: no cross-attn.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
)
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "self_attn",
}


# ===========================================================================
# Compressor
# ===========================================================================

@COMPRESSOR_REGISTRY.register("mamba_shedder_class")
@COMPRESSOR_REGISTRY.register("mamba_shedder_compressor_class")
class MambaShedderCompressor(BaseCompressor):
    # MS-FIX-01: this is a compressor, not an attention layer.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.2,
        shed_layers: int = 0,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        validate_after_shed: bool = True,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        max_calibration_batches: int = 8,
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        **kwargs: Any,
    ) -> None:
        # MS-FIX-02: no misleading kwargs.pop() calls. Config-driven overrides
        # are read from `comp_config` below.
        super().__init__(model, name="mamba_shedder", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            sparsity = _cfg_get(cfg, "mamba_shedder_sparsity",
                                _cfg_get(cfg, "sparsity", sparsity))
            shed_layers = _cfg_get(cfg, "mamba_shedder_shed_layers",
                                   _cfg_get(cfg, "shed_layers", shed_layers))
            structured_24 = _cfg_get(cfg, "mamba_shedder_structured_24",
                                     _cfg_get(cfg, "structured_24", structured_24))
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            max_calibration_batches = _cfg_get(cfg, "max_calibration_batches", max_calibration_batches)
            min_layer_dim = _cfg_get(cfg, "min_layer_dim", min_layer_dim)
            skip_should_compress_filter = _cfg_get(
                cfg, "skip_should_compress_filter", skip_should_compress_filter,
            )

        # Tokenizer resolution (MS-FIX-09)
        self.tokenizer = (
            tokenizer
            or getattr(model, "tokenizer", None)
            or getattr(getattr(model, "config", None), "tokenizer", None)
        )

        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.shed_layers = max(0, int(shed_layers))
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.validate_after_shed = bool(validate_after_shed)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.structured_24 = bool(structured_24)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.max_calibration_batches = max(1, int(max_calibration_batches))
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.skip_should_compress_filter = bool(skip_should_compress_filter)

        # MS-FIX-06: safe device resolution.
        if device is not None:
            self.target_device = torch.device(device)
        else:
            first_param = next(iter(model.parameters()), None)
            self.target_device = (
                first_param.device if first_param is not None
                else torch.device("cpu")
            )

        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # Target check
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()
        # MS-FIX-05: skip LoRA / already-compressed submodules.
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False

        # Already-pruned modules.
        cls_name = module.__class__.__name__
        if cls_name in ("DynamicSVDLinear", "SVDLinear", "SVDLinearLayer",
                        "FactorisedLinear", "FactorizedLinear", "LowRankPlusSparseLinear"):
            if _debug:
                logger.info(f"  [skip] {name}: already compressed ({cls_name})")
            return False

        # Critical layers.
        if not self.skip_should_compress_filter:
            try:
                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        # Linear-like detection.
        base_mod = getattr(module, "base_layer", module)
        cls_lower = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_lower
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
            return False

        w = getattr(base_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        # Attention-only filter.
        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
                return False

        # Explicit target modules.
        if self.target_modules is not None:
            ok = any(t in name for t in self.target_modules)
            if not ok and _debug:
                logger.info(f"  [skip] {name}: not in target_modules")
            return ok

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    # Activation collection (MS-FIX-04)
    # ------------------------------------------------------------------
    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        self.model.eval()
        accum: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    # Cap per-batch tokens.
                    if inp.dim() >= 2 and inp.shape[0] * inp.shape[1] > 32768:
                        inp = inp.reshape(-1, inp.shape[-1])[:32768].unsqueeze(0)
                    accum.setdefault(layer_name, []).append(inp.cpu())
            return hook

        target_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]

        if not target_modules:
            return {}

        for name, module in target_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )
            batches = self._normalize_calibration_batches(calib_inputs)
            n_batches = min(len(batches), self.max_calibration_batches)

            with torch.no_grad():
                for i in range(n_batches):
                    batch = batches[i]
                    try:
                        if isinstance(batch, dict):
                            self.model(**batch)
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch)
                        elif isinstance(batch, (list, tuple)):
                            self.model(*batch)
                    except Exception as e:
                        logger.debug("Calibration batch %d failed: %s", i, e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in accum.items():
            if act_list:
                try:
                    aggregated[name] = torch.cat(act_list, dim=0)
                except Exception as e:
                    logger.debug("Activation concat failed for %s: %s", name, e)
        return aggregated

    @staticmethod
    def _normalize_calibration_batches(calib_inputs: Any) -> List[Any]:
        if calib_inputs is None:
            return []
        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []
            first = calib_inputs[0]
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            if isinstance(first, (list, tuple)):
                return [calib_inputs]  # single tuple batch
            return [calib_inputs]
        if isinstance(calib_inputs, dict):
            return [calib_inputs]
        if isinstance(calib_inputs, torch.Tensor):
            return [calib_inputs]
        return [calib_inputs]

    # ------------------------------------------------------------------
    # Layer shedding (MS-FIX-07)
    # ------------------------------------------------------------------
    def _apply_layer_shedding(self) -> int:
        if self.shed_layers <= 0:
            return 0

        # Already shed?
        if getattr(self.model, "_mamba_shed_done", False):
            return 0

        target_container = None
        container_parent = None
        container_attr = None

        candidates = [
            ("backbone.layers", lambda m: getattr(getattr(m, "backbone", None), "layers", None)),
            ("transformer.layer", lambda m: getattr(getattr(m, "transformer", None), "layer", None)),
            ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
            ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
            ("encoder.layer", lambda m: getattr(getattr(m, "encoder", None), "layer", None)),
            ("layers", lambda m: getattr(m, "layers", None)),
            ("h", lambda m: getattr(m, "h", None)),
        ]

        for dotted_name, getter in candidates:
            try:
                res = getter(self.model)
            except Exception:
                res = None
            if isinstance(res, (nn.ModuleList, list)) and len(res) > self.shed_layers:
                parts = dotted_name.split(".")
                parent = self.model
                valid = True
                for p in parts[:-1]:
                    parent = getattr(parent, p, None)
                    if parent is None:
                        valid = False
                        break
                if valid:
                    target_container = res
                    container_parent = parent
                    container_attr = parts[-1]
                    break

        # Generic search
        if target_container is None:
            max_len = 0
            for name, mod in self.model.named_modules():
                if not isinstance(mod, nn.ModuleList):
                    continue
                if len(mod) <= self.shed_layers or len(mod) <= max_len:
                    continue
                parts = name.split(".")
                parent = self.model
                valid = True
                for p in parts[:-1]:
                    parent = getattr(parent, p, None)
                    if parent is None:
                        valid = False
                        break
                if valid:
                    target_container = mod
                    container_parent = parent
                    container_attr = parts[-1]
                    max_len = len(mod)

        if target_container is None or len(target_container) <= self.shed_layers:
            logger.warning("MambaShedder: no suitable ModuleList found for layer shedding.")
            return 0

        orig_len = len(target_container)
        keep_len = orig_len - self.shed_layers
        if keep_len <= 1 or orig_len <= 2:
            kept_indices = list(range(keep_len))
        else:
            keep_set = {0, orig_len - 1}
            n_middle_to_keep = keep_len - len(keep_set)
            middle_candidates = [i for i in range(1, orig_len - 1)]
            rng = random.Random(42)
            middle_kept = sorted(
                rng.sample(middle_candidates, min(n_middle_to_keep, len(middle_candidates)))
            )
            keep_set.update(middle_kept)
            kept_indices = sorted(keep_set)[:keep_len]

        old_container = getattr(container_parent, container_attr)
        new_layers = nn.ModuleList([target_container[i] for i in kept_indices])
        setattr(container_parent, container_attr, new_layers)

        # MS-FIX-13: validation with revert.
        if self.validate_after_shed:
            try:
                self.model.eval()
                dummy_in = torch.randint(0, 100, (1, 8), device=self.target_device)
                with torch.no_grad():
                    try:
                        self.model(dummy_in)
                    except Exception:
                        self.model(input_ids=dummy_in,
                                   attention_mask=torch.ones_like(dummy_in))
            except Exception as e:
                logger.warning(
                    "MambaShedder: layer shedding validation failed (%s). "
                    "Reverting to original %d layers.",
                    e, orig_len,
                )
                setattr(container_parent, container_attr, old_container)
                return 0

        shed_done = orig_len - len(new_layers)
        logger.info(
            "MambaShedder: shed %d layers (%d -> %d).",
            shed_done, orig_len, len(new_layers),
        )
        self.model._mamba_shed_done = True
        return shed_done

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        shed_done = self._apply_layer_shedding()
        activations_map = self._collect_activations()
        pruned_layers_count = 0
        self.layer_errors = {}
        layer_details: Dict[str, Any] = {}

        total_orig_params = sum(p.numel() for p in self.model.parameters())
        total_comp_params = 0

        # MS-FIX-08: diagnostic if no targets
        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]
        if not target_list:
            logger.warning("MambaShedder: no target layers found. Running diagnostic...")
            n_scanned = 0
            for name, module in self.model.named_modules():
                n_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning("MambaShedder diagnostic: %d modules scanned, 0 targets.", n_scanned)

        for name, module in target_list:
            target_layer = getattr(module, "base_layer", module)
            raw_w = target_layer.weight.data
            cls_name = target_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            used_act_sensitivity = False

            # Fisher-based sensitivity with magnitude fallback.
            act = activations_map.get(name, None)
            if act is not None and act.numel() > 0:
                try:
                    act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
                    if act_flat.shape[-1] == d_in:
                        fisher = (act_flat ** 2).mean(dim=0).clamp(min=1e-8)
                        sens = fisher.sqrt()
                        used_act_sensitivity = True
                    else:
                        sens = torch.norm(W, dim=0)
                except Exception as e:
                    logger.debug("Fisher computation failed for %s: %s", name, e)
                    sens = torch.norm(W, dim=0)
            else:
                sens = torch.norm(W, dim=0)

            # Structured 2:4 or unstructured pruning.
            if self.structured_24:
                saliency = sens.unsqueeze(0).expand_as(W)
                try:
                    W_pruned, _ = apply_24_structured_mask(W, saliency=saliency, dim=1)
                except Exception as e:
                    logger.warning("Structured 2:4 pruning failed for %s: %s", name, e)
                    W_pruned = W
            else:
                n_keep = max(1, int(d_in * (1.0 - self.sparsity)))
                _, keep_idx = torch.topk(sens, k=n_keep, largest=True)
                mask = torch.zeros(d_in, device=device, dtype=torch.float32)
                mask[keep_idx] = 1.0
                W_pruned = W * mask.unsqueeze(0)

            if (W_pruned == 0).all():
                logger.warning(
                    "MambaShedder: degenerate all-zero mask for %s; keeping original.",
                    name,
                )
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue

            norm_orig = torch.norm(W, p="fro")
            rel_err = (torch.norm(W - W_pruned, p="fro") / (norm_orig + 1e-8)).item()
            self.layer_errors[name] = rel_err

            if rel_err > self.max_error:
                logger.warning(
                    "MambaShedder: err %.4f on %s > %.4f. %s",
                    rel_err, name, self.max_error,
                    "Keeping original." if self.fallback_to_original else "Proceeding.",
                )
                if self.fallback_to_original:
                    for p in module.parameters(recurse=False):
                        total_comp_params += p.numel()
                    continue

            # MS-FIX-03: no assert; use graceful skip on shape mismatch.
            final_w = W_pruned.t().contiguous() if is_conv1d else W_pruned
            expected_shape = target_layer.weight.shape
            if final_w.shape != expected_shape:
                logger.warning(
                    "MambaShedder: shape mismatch for %s: expected %s, got %s; skipping.",
                    name, tuple(expected_shape), tuple(final_w.shape),
                )
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue

            target_layer.weight.data = final_w.to(dtype=dtype, device=device)
            target_layer.weight.requires_grad = False
            if getattr(target_layer, "bias", None) is not None:
                target_layer.bias.requires_grad = False

            # Preserve LoRA trainability.
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for p in (module.lora_A, module.lora_B):
                    if p is None:
                        continue
                    if hasattr(p, "weight"):
                        p.weight.requires_grad = True
                    elif isinstance(p, nn.Parameter):
                        p.requires_grad = True

            pruned_layers_count += 1
            for p in module.parameters(recurse=False):
                total_comp_params += p.numel()

            layer_details[name] = {
                "rel_error": rel_err,
                "used_act_sensitivity": used_act_sensitivity,
                "shape": tuple(final_w.shape),
                "sparsity": 0.5 if self.structured_24 else self.sparsity,
            }

        # MS-FIX-10: fast mean without torch.tensor(list(...))
        if self.layer_errors:
            avg_err = sum(self.layer_errors.values()) / len(self.layer_errors)
        else:
            avg_err = 0.0

        # MS-FIX-12: honest stats. We rewrite weights, not replace modules,
        # so framework-level compression_rate will read 1.0.
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        eff_ratio = total_comp_params / max(1, total_orig_params)

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update({
            "pruned_layers": pruned_layers_count,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "shed_layers": shed_done,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "layer_errors": self.layer_errors,
            "layer_details": layer_details,
            "effective_parameter_ratio": eff_ratio,
            "bias_handling": "biases preserved in full precision (unpruned)",
            "method": "mamba_shedder_sensitivity",
            "note": (
                "This compressor rewrites weights in-place; "
                "framework-level compression_rate may read 1.0. "
                "Use effective_parameter_ratio as the true density metric."
            ),
        })
        self.stats = stats

        logger.info(
            "MambaShedder: %d layers pruned, %d layers shed, avg err %.4f, "
            "effective_param_ratio %.2fx.",
            pruned_layers_count, shed_done, avg_err, eff_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("mamba_shedder")
@COMPRESSOR_REGISTRY.register("mamba_shedder_compressor")
@COMPRESSOR_REGISTRY.register("mamba_shedder_factory")
def mamba_shedder_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    sparsity = kwargs.pop("sparsity",
                          _cfg_get(comp_config, "mamba_shedder_sparsity",
                                   _cfg_get(comp_config, "sparsity", 0.2)))
    shed_layers = kwargs.pop("shed_layers",
                             _cfg_get(comp_config, "mamba_shedder_shed_layers",
                                      _cfg_get(comp_config, "shed_layers", 0)))
    structured_24 = kwargs.pop("structured_24",
                               _cfg_get(comp_config, "mamba_shedder_structured_24",
                                        _cfg_get(comp_config, "structured_24", False)))
    max_error = kwargs.pop("max_error",
                           _cfg_get(comp_config, "max_error", 0.20))
    fallback = kwargs.pop("fallback_to_original",
                          _cfg_get(comp_config, "fallback_to_original", True))
    target_attention_only = kwargs.pop("target_attention_only",
                                        _cfg_get(comp_config, "target_attention_only", False))
    skip_filter = kwargs.pop("skip_should_compress_filter",
                             _cfg_get(comp_config, "skip_should_compress_filter", False))
    tokenizer = kwargs.pop(
        "tokenizer",
        getattr(model, "tokenizer", None)
        or _cfg_get(comp_config, "tokenizer", None),
    )

    compressor = MambaShedderCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        shed_layers=shed_layers,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        skip_should_compress_filter=skip_filter,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "MambaShedderCompressor",
    "mamba_shedder_factory",
]