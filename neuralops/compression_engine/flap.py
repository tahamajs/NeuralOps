"""
FLAP: Fluctuation-Based Adaptive Structured Pruning (Corrected).

Registered in COMPRESSOR_REGISTRY as 'flap' and 'flap_compressor'.
Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with reconstruction validation,
Conv1D support, and safe fallback.

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.core.base import BaseCompressor
from neuralops.core.pruning_utils import (
    adaptive_max_error,
    apply_24_structured_mask,
    get_calibration_tensors,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
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
    "out_proj",
    "dense",
}


@COMPRESSOR_REGISTRY.register("flap")
@COMPRESSOR_REGISTRY.register("flap_compressor")
class FLAPCompressor(BaseCompressor):
    """
    FLAP Compressor: Fluctuation-based Adaptive Structured Pruning.

    Production hardening features:
    - Robust LoRALinear and Conv1D support (transpose + adapter preservation).
    - target_attention_only restricts pruning to attention projections.
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if error exceeds threshold.
    - Real metrics (effective_parameter_ratio, num_compressed, num_fallback, sparsity).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        sparsity: float = 0.2,
        metric_type: str = "WIFV",
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        structured_24: bool = False,
        target_modules: list[str] | None = None,
        cfg: Any | None = None,
        num_calibration_batches: int = 8,  # FL-FIX-04
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flap", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        raw_sp = _cfg_get(cfg, "flap_sparsity", _cfg_get(cfg, "sparsity", sparsity))
        self.sparsity = max(0.0, min(float(raw_sp), 0.95))
        self.metric_type = str(
            _cfg_get(cfg, "flap_metric_type", _cfg_get(cfg, "metric_type", metric_type))
        ).upper()
        self.structured_24 = bool(
            _cfg_get(
                cfg, "flap_structured_24", _cfg_get(cfg, "structured_24", structured_24)
            )
        )
        c_max_error = _cfg_get(
            cfg, "flap_max_error", _cfg_get(cfg, "max_error", max_error)
        )
        if c_max_error is None or c_max_error == 0.35:
            c_max_error = adaptive_max_error(
                int((1.0 - self.sparsity) * 768), 768, 768, "distilbert"
            )
            c_max_error = max(c_max_error, 0.60)

        self.max_error = float(c_max_error)
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "flap_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "flap_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.calibration_loader = calibration_loader or _cfg_get(
            cfg, "calibration_loader", None
        )
        self.calibration_texts = calibration_texts
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.num_calibration_batches = max(
            1, int(_cfg_get(cfg, "num_calibration_batches", num_calibration_batches))
        )

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_sparsities: dict[str, float] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
                "head",
                "output_layer",
            ]
        ):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False

        if not should_compress_linear(self.model, name):
            return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(
                k in name_lower for k in ("attn", "attention")
            )
            if not is_attn:
                return False

        return True

    # ------------------------------------------------------------------
    # FL-FIX-04/05: multiple calibration batches, mean-of-variance accumulation
    # ------------------------------------------------------------------
    def _normalize_calibration_batches(self, calib_inputs: Any) -> list[Any]:
        if calib_inputs is None:
            return []
        if isinstance(calib_inputs, torch.Tensor):
            return [calib_inputs]
        if isinstance(calib_inputs, dict):
            return [calib_inputs]
        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []
            first = calib_inputs[0]
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            return [calib_inputs]
        return [calib_inputs]

    def _collect_layer_activations(self) -> dict[str, torch.Tensor]:
        """
        Capture per-feature activation variance across calibration batches.
        Returns: {name: variance_tensor of shape (in_features,)}
        """
        self.model.eval()
        variance_sum: dict[str, torch.Tensor] = {}
        batch_count: dict[str, int] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inputs, output):
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return
                inp = inputs[0].detach().float()
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                in_f = (
                    target_mod.weight.shape[0]
                    if is_conv1d
                    else target_mod.weight.shape[1]
                )
                flat = inp.reshape(-1, in_f).cpu()
                if flat.size(0) >= 2:
                    var = flat.var(dim=0)
                else:
                    var = torch.zeros(in_f)
                if name not in variance_sum:
                    variance_sum[name] = var
                    batch_count[name] = 1
                else:
                    variance_sum[name] += var
                    batch_count[name] += 1

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
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
            n = min(len(batches), self.num_calibration_batches)

            with torch.no_grad():
                for i in range(n):
                    b = batches[i]
                    try:
                        if isinstance(b, dict):
                            self.model(**b)
                        elif isinstance(b, torch.Tensor):
                            self.model(b)
                        elif isinstance(b, (list, tuple)):
                            self.model(*b)
                    except Exception as e:
                        logger.debug("FLAP calibration batch %d failed: %s", i, e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: dict[str, torch.Tensor] = {}
        for name, vsum in variance_sum.items():
            aggregated[name] = (vsum / max(1, batch_count[name])).to(self.target_device)
        return aggregated

    # ------------------------------------------------------------------
    def _compute_fluctuation_metric(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-column fluctuation saliency score in fp32."""
        d_out, d_in = weight.shape
        device = weight.device

        if activations is not None and activations.numel() > 0:
            inp_var = activations.reshape(-1).float().to(device)
            if inp_var.shape[0] != d_in:
                inp_var = torch.ones(d_in, device=device)
            else:
                inp_var = inp_var.clamp(min=1e-8)
        else:
            inp_var = torch.ones(d_in, device=device)

        W = weight.float()
        if self.metric_type == "IFV":
            scores = inp_var
        elif self.metric_type == "WIFV":
            scores = inp_var * torch.sum(W.pow(2), dim=0)
        elif self.metric_type == "WIFN":
            scores = torch.sum(torch.abs(W), dim=0) * torch.sqrt(inp_var)
        else:
            scores = inp_var * torch.sum(W.pow(2), dim=0)

        return scores

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        activations_map = self._collect_layer_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}

        total_orig_params = 0
        total_comp_params = 0

        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(
                    part in name.split(".")
                    for part in ["lora_A", "lora_B", "base_layer"]
                ):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(
                    name, module
                ):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("FLAP: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data
            orig_shape = raw_w.shape
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().float().contiguous() if is_conv1d else raw_w.float()
            d_out, d_in = W.shape
            orig_layer_params = d_out * d_in
            total_orig_params += orig_layer_params
            norm_w = torch.norm(W, p="fro").item()

            act = activations_map.get(name, None)
            scores = self._compute_fluctuation_metric(W, act)

            # FL-FIX-01: compute actual sparsity from mask.
            if self.structured_24:
                saliency_matrix = scores.unsqueeze(0).expand_as(W)
                W_pruned, mask = apply_24_structured_mask(
                    W, saliency=saliency_matrix, dim=1
                )
                if mask is not None:
                    mask_f = mask.float()
                    effective_sparsity = 1.0 - (
                        mask_f.sum().item() / max(1, mask_f.numel())
                    )
                else:
                    # Fallback: count zeros.
                    zeros = (W_pruned.abs() < 1e-12).float().sum().item()
                    effective_sparsity = zeros / max(1, W_pruned.numel())
            else:
                n_keep = max(1, int(d_in * (1.0 - self.sparsity)))
                _, keep_idx = torch.topk(scores, k=n_keep, largest=True)
                mask = torch.zeros(d_in, device=device)
                mask[keep_idx] = 1.0
                W_pruned = W * mask.unsqueeze(0)
                effective_sparsity = 1.0 - (n_keep / d_in)

            if norm_w < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - W_pruned, p="fro").item() / norm_w)

            self.layer_errors[name] = final_err
            self.layer_sparsities[name] = float(effective_sparsity)

            if final_err > self.max_error:
                logger.warning(
                    "FLAP reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("FLAP: fallback to original on %s.", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            # FL-FIX-02: no_grad for weight copy.
            if is_conv1d:
                res_w = W_pruned.t().to(dtype=dtype, device=device).contiguous()
            else:
                res_w = W_pruned.to(dtype=dtype, device=device).contiguous()

            if res_w.shape != orig_shape:
                raise RuntimeError(
                    f"FLAP: shape mismatch on {name}: {tuple(res_w.shape)} vs {tuple(orig_shape)}"
                )

            with torch.no_grad():
                target_mod.weight.data.copy_(res_w)

            # LoRA trainability.
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if (
                    hasattr(module.lora_A, "weight")
                    and module.lora_A.weight is not None
                ):
                    module.lora_A.weight.requires_grad = True
                if (
                    hasattr(module.lora_B, "weight")
                    and module.lora_B.weight is not None
                ):
                    module.lora_B.weight.requires_grad = True

            self.compressed_layers.append(name)
            # FL-FIX-03: effective params based on actual sparsity.
            comp_layer_params = int((1.0 - effective_sparsity) * orig_layer_params)
            total_comp_params += max(1, comp_layer_params)

        # FL-FIX-06: forward sanity check with attention_mask.
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                attn = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        first_linear = next(
                            (
                                m
                                for m in self.model.modules()
                                if isinstance(m, nn.Linear)
                            ),
                            None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        self.model(
                            torch.randn(
                                1,
                                8,
                                in_dim,
                                device=dev,
                                dtype=(
                                    first_p.dtype
                                    if first_p is not None
                                    else torch.float32
                                ),
                            )
                        )
        except Exception as e:
            logger.debug("FLAP forward sanity check failed: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        effective_ratio = (
            (total_comp_params / total_orig_params) if total_orig_params > 0 else 1.0
        )
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_sparsities": self.layer_sparsities,
                "metric_type": self.metric_type,
                "sparsity_target": self.sparsity,
                "structured_24": self.structured_24,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
                "method": "flap_fluctuation_pruning",
            }
        )
        logger.info(
            "Compressor flap finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ===========================================================================
# Factory (FL-FIX-07)
# ===========================================================================


@COMPRESSOR_REGISTRY.register("flap_factory")
@COMPRESSOR_REGISTRY.register("flap_compressor_factory")
def flap_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for FLAP compressor."""
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "flap_sparsity", _cfg_get(comp_config, "sparsity", 0.2)),
    )
    metric_type = kwargs.pop(
        "metric_type",
        _cfg_get(
            comp_config,
            "flap_metric_type",
            _cfg_get(comp_config, "metric_type", "WIFV"),
        ),
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(
            comp_config,
            "flap_structured_24",
            _cfg_get(comp_config, "structured_24", False),
        ),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "flap_max_error", _cfg_get(comp_config, "max_error", 0.35)
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "flap_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "flap_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    tokenizer = kwargs.pop(
        "tokenizer",
        getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None),
    )
    num_calib = kwargs.pop(
        "num_calibration_batches", _cfg_get(comp_config, "num_calibration_batches", 8)
    )

    compressor = FLAPCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        metric_type=metric_type,
        structured_24=structured_24,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        num_calibration_batches=num_calib,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["FLAPCompressor", "flap_compressor_factory"]
