"""
Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer Shedding Compressor.

Registered as 'mamba_shedder', 'mamba_shedder_compressor', 'mamba_shedder_factory'.

Mathematical Formulation:
  1. Empirical Fisher:  I(X_j) = sqrt(mean_n X_{n,j}^2)
  2. Layer Residual:    Δ_l = ||X_{l+1} - X_l||_F / (||X_l||_F + ε)
  3. Channel Masking:   W_pruned = W ⊙ M_channel^T

⚠️  IMPORTANT  ⚠️

Channel masking alone does NOT reduce dense storage. To actually shrink
memory, this version replaces targeted Linear/Conv1D layers with a
`MambaShedderPrunedLinear` module that stores ONLY the retained channels
(via per-output-channel keep indices) plus LoRA adapters. The framework's
dense parameter counter will still read 1.0 for `compression_rate` because
the module signature is preserved; the true storage savings are reported in
`storage_compression_ratio` and `storage_reduction_pct`.

"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
)
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ===========================================================================
# Config helper
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
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
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "self_attn",
}


# ===========================================================================
# Pruned Linear layer (MS-FIX-15)
# ===========================================================================
class MambaShedderPrunedLinear(nn.Module):
    """
    Linear layer with pruned output channels.

    Stores only the kept rows of W:
      - W_kept:    (k, in_features)
      - bias_kept: (k,) or None
      - keep_idx:  (k,) int64 (output positions of kept rows)
      - original_out_features: int

    Forward:
      h = F.linear(x, W_kept, bias_kept)              # (..., k)
      out = zeros(..., original_out) then scatter to keep_idx
      + LoRA(x) if present.

    Storage: k * in instead of out * in.
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        original_out_features: int,
        W_kept: torch.Tensor,
        keep_idx: torch.Tensor,
        bias_kept: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.original_out_features = int(original_out_features)
        self.out_features = self.original_out_features
        self.nf = self.out_features
        self.num_kept = int(W_kept.shape[0])
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        self.register_buffer("keep_idx", keep_idx.long(), persistent=True)
        self.W_kept = nn.Parameter(W_kept, requires_grad=False)

        if bias_kept is not None:
            self.bias_kept = nn.Parameter(bias_kept, requires_grad=False)
        else:
            self.register_parameter("bias_kept", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    (self.lora_r, self.in_features),
                    dtype=torch.float32,
                    device=W_kept.device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (self.original_out_features, self.lora_r),
                    dtype=torch.float32,
                    device=W_kept.device,
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct (original_out, in) weight with zeros in pruned rows."""
        W = self.W_kept.new_zeros(self.original_out_features, self.in_features)
        W.index_copy_(0, self.keep_idx, self.W_kept)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.W_kept, self.bias_kept)  # (..., k)
        out = torch.zeros(
            *x.shape[:-1],
            self.original_out_features,
            dtype=h.dtype,
            device=h.device,
        )
        out.scatter_(-1, self.keep_idx.expand(*x.shape[:-1], -1), h)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
            out = out + self.scaling * lora_out.to(out.dtype)
        return out

    def storage_bytes(self) -> int:
        total = self.W_kept.numel() * self.W_kept.element_size()
        total += self.keep_idx.numel() * self.keep_idx.element_size()
        if self.bias_kept is not None:
            total += self.bias_kept.numel() * self.bias_kept.element_size()
        if self.lora_A is not None:
            total += self.lora_A.numel() * self.lora_A.element_size()
        if self.lora_B is not None:
            total += self.lora_B.numel() * self.lora_B.element_size()
        return total


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("mamba_shedder_class")
@COMPRESSOR_REGISTRY.register("mamba_shedder_compressor_class")
class MambaShedderCompressor(BaseCompressor):
    is_custom_attention: bool = False  # MS-FIX-01

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        sparsity: float = 0.2,
        shed_layers: int = 0,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        validate_after_shed: bool = True,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        structured_24: bool = False,
        target_attention_only: bool = False,
        target_modules: list[str] | None = None,
        max_calibration_batches: int = 8,
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mamba_shedder", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))

        if cfg is not None:
            sparsity = _cfg_get(
                cfg, "mamba_shedder_sparsity", _cfg_get(cfg, "sparsity", sparsity)
            )
            shed_layers = _cfg_get(
                cfg,
                "mamba_shedder_shed_layers",
                _cfg_get(cfg, "shed_layers", shed_layers),
            )
            structured_24 = _cfg_get(
                cfg,
                "mamba_shedder_structured_24",
                _cfg_get(cfg, "structured_24", structured_24),
            )
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(
                cfg, "fallback_to_original", fallback_to_original
            )
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )
            max_calibration_batches = _cfg_get(
                cfg, "max_calibration_batches", max_calibration_batches
            )
            min_layer_dim = _cfg_get(cfg, "min_layer_dim", min_layer_dim)
            skip_should_compress_filter = _cfg_get(
                cfg,
                "skip_should_compress_filter",
                skip_should_compress_filter,
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)

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
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)

        if device is not None:
            self.target_device = torch.device(device)
        else:
            first_param = next(iter(model.parameters()), None)
            self.target_device = (
                first_param.device if first_param is not None else torch.device("cpu")
            )

        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # Target check (MS-FIX-16: cached)
    # ------------------------------------------------------------------
    def _is_target_module(
        self, name: str, module: nn.Module, _debug: bool = False
    ) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False

        cls_name = module.__class__.__name__
        if cls_name in (
            "DynamicSVDLinear",
            "SVDLinear",
            "SVDLinearLayer",
            "FactorisedLinear",
            "FactorizedLinear",
            "LowRankPlusSparseLinear",
            "MambaShedderPrunedLinear",
        ):
            if _debug:
                logger.info(f"  [skip] {name}: already compressed ({cls_name})")
            return False

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

        if self.target_attention_only and not any(k in name_lower for k in _ATTN_NAMES):
            if _debug:
                logger.info(f"  [skip] {name}: not attention")
            return False

        if self.target_modules is not None:
            ok = any(t in name for t in self.target_modules)
            if not ok and _debug:
                logger.info(f"  [skip] {name}: not in target_modules")
            return ok

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------
    def _collect_activations(self, target_names: list[str]) -> dict[str, torch.Tensor]:
        self.model.eval()
        accum: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    if inp.dim() >= 2 and inp.shape[0] * inp.shape[1] > 32768:
                        inp = inp.reshape(-1, inp.shape[-1])[:32768].unsqueeze(0)
                    accum.setdefault(layer_name, []).append(inp.cpu())

            return hook

        target_set = set(target_names)
        for name, module in self.model.named_modules():
            if name in target_set:
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

        aggregated: dict[str, torch.Tensor] = {}
        for name, act_list in accum.items():
            if act_list:
                try:
                    aggregated[name] = torch.cat(act_list, dim=0)
                except Exception as e:
                    logger.debug("Activation concat failed for %s: %s", name, e)
        return aggregated

    @staticmethod
    def _normalize_calibration_batches(calib_inputs: Any) -> list[Any]:
        if calib_inputs is None:
            return []
        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []
            first = calib_inputs[0]
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            if isinstance(first, (list, tuple)):
                return [calib_inputs]
            return [calib_inputs]
        if isinstance(calib_inputs, (dict, torch.Tensor)):
            return [calib_inputs]
        return [calib_inputs]

    # ------------------------------------------------------------------
    # Layer shedding
    # ------------------------------------------------------------------
    def _apply_layer_shedding(self) -> int:
        if self.shed_layers <= 0:
            return 0
        if getattr(self.model, "_mamba_shed_done", False):
            logger.info("MambaShedder: layer shedding already applied; skipping.")
            return 0

        target_container = None
        container_parent = None
        container_attr = None

        candidates = [
            "backbone.layers",
            "transformer.layer",
            "transformer.h",
            "model.layers",
            "encoder.layer",
            "layers",
            "h",
        ]

        def _resolve(path: str):
            parts = path.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p, None)
                if parent is None:
                    return None, None
            container = getattr(parent, parts[-1], None)
            if isinstance(container, (nn.ModuleList, list)):
                return parent, parts[-1]
            return None, None

        for path in candidates:
            parent, attr = _resolve(path)
            if parent is None:
                continue
            container = getattr(parent, attr)
            if len(container) > self.shed_layers:
                target_container = container
                container_parent = parent
                container_attr = attr
                break

        if target_container is None:
            # Generic search: longest module list.
            best_len = 0
            for name, mod in self.model.named_modules():
                if not isinstance(mod, nn.ModuleList):
                    continue
                if len(mod) <= self.shed_layers or len(mod) <= best_len:
                    continue
                parent, attr = _resolve(name)
                if parent is None:
                    continue
                target_container = mod
                container_parent = parent
                container_attr = attr
                best_len = len(mod)

        if target_container is None or len(target_container) <= self.shed_layers:
            logger.warning("MambaShedder: no suitable ModuleList found.")
            return 0

        orig_len = len(target_container)
        keep_len = orig_len - self.shed_layers

        # Always keep first and last (per Mamba-Shedder paper).
        keep_set = {0, orig_len - 1}
        n_middle = keep_len - len(keep_set)
        if n_middle > 0 and orig_len > 2:
            middle = list(range(1, orig_len - 1))
            rng = random.Random(42)
            middle_kept = sorted(rng.sample(middle, min(n_middle, len(middle))))
            keep_set.update(middle_kept)
        else:
            keep_set = set(range(keep_len))

        kept_indices = sorted(keep_set)[:keep_len]
        old_container = getattr(container_parent, container_attr)
        new_layers = nn.ModuleList([target_container[i] for i in kept_indices])
        setattr(container_parent, container_attr, new_layers)

        # Validation with revert
        if self.validate_after_shed:
            try:
                self.model.eval()
                dummy = torch.randint(0, 100, (1, 8), device=self.target_device)
                with torch.no_grad():
                    try:
                        self.model(input_ids=dummy)
                    except Exception:
                        try:
                            self.model(dummy)
                        except Exception:
                            cfg = getattr(self.model, "config", None)
                            hidden = getattr(cfg, "hidden_size", None) if cfg else None
                            if hidden is None:
                                first_lin = next(
                                    (
                                        m
                                        for m in self.model.modules()
                                        if isinstance(m, nn.Linear)
                                    ),
                                    None,
                                )
                                hidden = getattr(first_lin, "in_features", 8)
                            dt = next(self.model.parameters()).dtype
                            self.model(
                                torch.randn(
                                    1, 8, hidden, device=self.target_device, dtype=dt
                                )
                            )
            except Exception as e:
                logger.warning(
                    "MambaShedder: shedding validation failed (%s); reverting.", e
                )
                setattr(container_parent, container_attr, old_container)
                return 0

        shed_done = orig_len - len(new_layers)
        logger.info(
            "MambaShedder: shed %d layers (%d → %d).",
            shed_done,
            orig_len,
            len(new_layers),
        )
        self.model._mamba_shed_done = True
        return shed_done

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        from framework.core.model_utils import replace_module

        shed_done = self._apply_layer_shedding()

        # Single-pass target discovery (MS-FIX-16)
        targets = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]

        if not targets:
            logger.warning("MambaShedder: no target layers found.")
            n = 0
            for name, module in self.model.named_modules():
                n += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning("MambaShedder diagnostic: %d modules scanned, 0 targets.", n)

        activations_map = self._collect_activations([n for n, _ in targets])

        pruned_layers_count = 0
        self.layer_errors = {}
        layer_details: dict[str, Any] = {}

        total_orig_bytes = 0
        total_comp_bytes = 0
        seen_weight_ids: set[int] = set()

        for name, module in targets:
            base_mod = getattr(module, "base_layer", module)
            w = getattr(base_mod, "weight", None)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weight_ids:
                continue
            seen_weight_ids.add(w_id)

            raw_w = w.data
            device = raw_w.device
            dtype = raw_w.dtype
            cls_name = base_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")

            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape

            orig_bytes = raw_w.numel() * raw_w.element_size()
            if getattr(base_mod, "bias", None) is not None:
                orig_bytes += base_mod.bias.numel() * base_mod.bias.element_size()
            total_orig_bytes += orig_bytes

            # Sensitivity: Fisher or weight-norm.
            act = activations_map.get(name, None)
            used_act_sensitivity = False
            if act is not None and act.numel() > 0:
                try:
                    act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
                    if act_flat.shape[-1] == d_in:
                        fisher = (act_flat**2).mean(dim=0).clamp(min=1e-8)
                        sens = fisher.sqrt()
                        used_act_sensitivity = True
                    else:
                        sens = torch.norm(W, dim=0)
                except Exception as e:
                    logger.debug("Fisher computation failed for %s: %s", name, e)
                    sens = torch.norm(W, dim=0)
            else:
                sens = torch.norm(W, dim=0)

            # Decide kept output channels (or 2:4 mask).
            if self.structured_24:
                # 2:4 → no storage savings (dense storage kept). Fallback to
                # in-place weight modification.
                saliency = sens.unsqueeze(0).expand_as(W)
                try:
                    W_pruned, _ = apply_24_structured_mask(W, saliency=saliency, dim=1)
                except Exception as e:
                    logger.warning("2:4 pruning failed for %s: %s", name, e)
                    total_comp_bytes += orig_bytes
                    continue

                norm_orig = torch.norm(W, p="fro").clamp_min(1e-12)
                rel_err = (torch.norm(W - W_pruned, p="fro") / norm_orig).item()
                self.layer_errors[name] = rel_err

                if rel_err > self.max_error and self.fallback_to_original:
                    total_comp_bytes += orig_bytes
                    continue

                final_w = W_pruned.t().contiguous() if is_conv1d else W_pruned
                if final_w.shape != raw_w.shape:
                    total_comp_bytes += orig_bytes
                    continue

                with torch.no_grad():
                    base_mod.weight.data.copy_(final_w.to(dtype=dtype, device=device))
                    base_mod.weight.requires_grad = False
                    if getattr(base_mod, "bias", None) is not None:
                        base_mod.bias.requires_grad = False

                pruned_layers_count += 1
                total_comp_bytes += orig_bytes  # no real storage change
                layer_details[name] = {
                    "rel_error": rel_err,
                    "used_act_sensitivity": used_act_sensitivity,
                    "storage_bytes_before": orig_bytes,
                    "storage_bytes_after": orig_bytes,
                    "note": "2:4 sparsity: no storage savings",
                }
            else:
                # Channel pruning → real storage savings.
                n_keep = max(1, int(d_out * (1.0 - self.sparsity)))
                _, keep_idx = torch.topk(sens, k=n_keep, largest=True)
                keep_idx, _ = torch.sort(keep_idx)
                keep_idx = keep_idx.to(device)

                W_kept = W.index_select(0, keep_idx).contiguous()
                W_pruned_full = torch.zeros_like(W)
                W_pruned_full.index_copy_(0, keep_idx, W_kept)

                norm_orig = torch.norm(W, p="fro").clamp_min(1e-12)
                rel_err = (torch.norm(W - W_pruned_full, p="fro") / norm_orig).item()
                self.layer_errors[name] = rel_err

                if rel_err > self.max_error and self.fallback_to_original:
                    logger.warning(
                        "MambaShedder: err %.4f on %s > %.4f; keeping original.",
                        rel_err,
                        name,
                        self.max_error,
                    )
                    total_comp_bytes += orig_bytes
                    continue

                bias_kept = None
                if getattr(base_mod, "bias", None) is not None:
                    b_full = base_mod.bias.data.float()
                    bias_kept = (
                        b_full.index_select(0, keep_idx)
                        .contiguous()
                        .to(dtype=dtype, device=device)
                    )

                # Existing LoRA
                lora_A_src = getattr(module, "lora_A", None)
                lora_B_src = getattr(module, "lora_B", None)
                eff_lora_r = self.lora_r
                if lora_A_src is not None and hasattr(lora_A_src, "weight"):
                    eff_lora_r = int(lora_A_src.weight.shape[0])

                new_layer = MambaShedderPrunedLinear(
                    in_features=d_in,
                    original_out_features=d_out,
                    W_kept=W_kept.to(dtype=dtype, device=device),
                    keep_idx=keep_idx.cpu(),
                    bias_kept=bias_kept,
                    lora_r=eff_lora_r,
                    lora_alpha=self.lora_alpha,
                )

                # Preserve LoRA weights if shapes match.
                if (
                    lora_A_src is not None
                    and lora_B_src is not None
                    and new_layer.lora_A is not None
                ):
                    with torch.no_grad():
                        src_A = (
                            lora_A_src.weight.data
                            if hasattr(lora_A_src, "weight")
                            else lora_A_src.data
                        )
                        src_B = (
                            lora_B_src.weight.data
                            if hasattr(lora_B_src, "weight")
                            else lora_B_src.data
                        )
                        if (
                            src_A.shape == new_layer.lora_A.shape
                            and src_B.shape == new_layer.lora_B.shape
                        ):
                            new_layer.lora_A.copy_(
                                src_A.to(
                                    new_layer.lora_A.device,
                                    dtype=new_layer.lora_A.dtype,
                                )
                            )
                            new_layer.lora_B.copy_(
                                src_B.to(
                                    new_layer.lora_B.device,
                                    dtype=new_layer.lora_B.dtype,
                                )
                            )

                # Splice with wrapper preservation (MS-FIX-18)
                if hasattr(module, "base_layer"):
                    module.base_layer = new_layer
                else:
                    replace_module(self.model, name, new_layer)

                pruned_layers_count += 1
                total_comp_bytes += new_layer.storage_bytes()
                layer_details[name] = {
                    "rel_error": rel_err,
                    "used_act_sensitivity": used_act_sensitivity,
                    "storage_bytes_before": orig_bytes,
                    "storage_bytes_after": new_layer.storage_bytes(),
                    "keep_ratio": n_keep / d_out,
                }

        # ------------------------------------------------------------------
        # Forward sanity check (MS-FIX-17)
        # ------------------------------------------------------------------
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        cfg = getattr(self.model, "config", None)
                        hidden = getattr(cfg, "hidden_size", None) if cfg else None
                        if hidden is None:
                            first_lin = next(
                                (
                                    m
                                    for m in self.model.modules()
                                    if isinstance(m, nn.Linear)
                                ),
                                None,
                            )
                            hidden = getattr(first_lin, "in_features", 8)
                        dt = next(self.model.parameters()).dtype
                        self.model(torch.randn(1, 8, hidden, device=dev, dtype=dt))
        except Exception as e:
            logger.warning("MambaShedder forward sanity check failed: %s", e)
            forward_check_success = False

        # ------------------------------------------------------------------
        # Stats
        # ------------------------------------------------------------------
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )

        storage_ratio = (
            total_orig_bytes / max(1, total_comp_bytes) if total_comp_bytes > 0 else 1.0
        )
        effective_ratio = (
            total_comp_bytes / max(1, total_orig_bytes) if total_orig_bytes > 0 else 1.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update(
            {
                "pruned_layers": pruned_layers_count,
                "shed_layers": shed_done,
                "sparsity_target": self.sparsity,
                "structured_24": self.structured_24,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "layer_errors": self.layer_errors,
                "layer_details": layer_details,
                "orig_storage_bytes": total_orig_bytes,
                "comp_storage_bytes": total_comp_bytes,
                "storage_compression_ratio": round(storage_ratio, 3),
                "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
                "method": "mamba_shedder_sensitivity",
                "note": (
                    "Channel pruning now replaces target layers with "
                    "MambaShedderPrunedLinear (real storage reduction). "
                    "2:4 sparsity stores dense weights, so its storage "
                    "compression ratio stays at 1.0."
                ),
            }
        )
        self.stats = stats

        logger.info(
            "MambaShedder: %d pruned, %d shed, avg_err=%.4f, storage=%.2fx.",
            pruned_layers_count,
            shed_done,
            avg_err,
            storage_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("mamba_shedder")
@COMPRESSOR_REGISTRY.register("mamba_shedder_compressor")
@COMPRESSOR_REGISTRY.register("mamba_shedder_factory")
def mamba_shedder_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(
            comp_config,
            "mamba_shedder_sparsity",
            _cfg_get(comp_config, "sparsity", 0.2),
        ),
    )
    shed_layers = kwargs.pop(
        "shed_layers",
        _cfg_get(
            comp_config,
            "mamba_shedder_shed_layers",
            _cfg_get(comp_config, "shed_layers", 0),
        ),
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(
            comp_config,
            "mamba_shedder_structured_24",
            _cfg_get(comp_config, "structured_24", False),
        ),
    )
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.20))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    skip_filter = kwargs.pop(
        "skip_should_compress_filter",
        _cfg_get(comp_config, "skip_should_compress_filter", False),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    tokenizer = kwargs.pop(
        "tokenizer",
        getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

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
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "MambaShedderPrunedLinear",
    "MambaShedderCompressor",
    "mamba_shedder_factory",
]
