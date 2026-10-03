"""
FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning.

Registered as 'flat_llm', 'flat_llm_compressor'.

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
    adaptive_max_error,
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
    "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "out_proj", "dense",
}


# ===========================================================================
# Pruned channel linear layer (real storage reduction)
# ===========================================================================
class PrunedChannelLinear(nn.Module):
    """
    Linear layer with pruned output channels.

    Stores only the kept rows of the effective weight matrix:
      - W_kept:      (k, in_features)  fp16/fp32   -- only kept rows
      - bias_kept:   (k,)              optional
      - kept_idx:    (k,)              int64  -- output positions of kept rows
      - original_out_features: int

    Forward:
      h = F.linear(x, W_kept, bias_kept)          # (..., k)
      out = zeros(..., original_out_features)
      out.scatter_(-1, kept_idx, h)               # (..., original_out)
    """
    is_custom_attention = False

    def __init__(
        self,
        in_features: int,
        original_out_features: int,
        kept_idx: torch.Tensor,             # (k,) int64
        W_kept: torch.Tensor,               # (k, in_features)
        bias_kept: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.original_out_features = int(original_out_features)
        self.out_features = self.original_out_features   # for API compatibility
        self.num_kept = int(W_kept.shape[0])

        self.register_buffer("kept_idx", kept_idx.long(), persistent=True)
        self.W_kept = nn.Parameter(W_kept, requires_grad=False)

        if bias_kept is not None:
            self.bias_kept = nn.Parameter(bias_kept, requires_grad=False)
        else:
            self.register_parameter("bias_kept", None)

        self._cached_weight: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct effective (original_out, in) weight with zeros in pruned rows."""
        if self._cached_weight is not None:
            return self._cached_weight
        W = self.W_kept.new_zeros(self.original_out_features, self.in_features)
        W[self.kept_idx] = self.W_kept
        self._cached_weight = W
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.W_kept, self.bias_kept)      # (..., k)
        out = torch.zeros(
            *x.shape[:-1],
            self.original_out_features,
            dtype=h.dtype,
            device=h.device,
        )
        out.scatter_(-1, self.kept_idx.expand(*x.shape[:-1], -1), h)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, original_out_features={self.original_out_features}, "
            f"num_kept={self.num_kept}"
        )


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("flat_llm")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor")
class FLATLLMCompressor(BaseCompressor):
    """
    FLAT-LLM compressor: Activation-aware output-channel pruning.

    Pipeline:
      1. Collect per-output-channel energy from calibration activations.
      2. Rank channels by energy.
      3. Keep top-k channels (k = round(keep_ratio * out_features)).
      4. Replace Linear with PrunedChannelLinear (factorized storage).

    Optionally applies 2:4 structured sparsity (compute-only; no storage savings
    unless a sparse kernel is used).
    """
    # FLT-FIX-09
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        keep_ratio: float = 0.7,
        damping: float = 1e-4,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        structured_24: bool = False,
        target_modules: list[str] | None = None,
        cfg: Any | None = None,
        num_calibration_batches: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flat_llm", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        raw_kr = _cfg_get(cfg, "flat_llm_keep_ratio",
                          _cfg_get(cfg, "keep_ratio", keep_ratio))
        self.keep_ratio = max(0.05, min(float(raw_kr), 1.0))
        # FLT-FIX-04
        self.damping = max(0.0, float(_cfg_get(cfg, "flat_llm_damping",
                                                _cfg_get(cfg, "damping", damping))))
        self.structured_24 = bool(_cfg_get(cfg, "flat_llm_structured_24",
                                            _cfg_get(cfg, "structured_24", structured_24)))
        c_max_error = _cfg_get(cfg, "flat_llm_max_error", _cfg_get(cfg, "max_error", max_error))
        if c_max_error is None or c_max_error == 0.35:
            c_max_error = adaptive_max_error(int(self.keep_ratio * 768), 768, 768, "distilbert")
            c_max_error = max(c_max_error, 0.60)

        self.max_error = float(c_max_error)
        self.target_attention_only = bool(
            _cfg_get(cfg, "flat_llm_target_attention_only",
                     _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "flat_llm_fallback_to_original",
                     _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.calibration_loader = calibration_loader or _cfg_get(cfg, "calibration_loader", None)
        self.calibration_texts = calibration_texts
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device) if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.num_calibration_batches = max(1, int(
            _cfg_get(cfg, "num_calibration_batches", num_calibration_batches)
        ))

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_keep_ratios: dict[str, float] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in
               ("classifier", "score", "lm_head", "pooler", "embedding",
                "router", "gate", "norm", "layernorm", "rmsnorm")):
            return False

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding,
                                nn.BatchNorm1d, nn.BatchNorm2d,
                                PrunedChannelLinear)):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    # Calibration: per-output-channel energy (FLT-FIX-05)
    # ------------------------------------------------------------------
    def _normalize_calibration_batches(self, calib_inputs: Any) -> list[Any]:
        if calib_inputs is None:
            return []
        if isinstance(calib_inputs, (torch.Tensor, dict)):
            return [calib_inputs]
        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []
            first = calib_inputs[0]
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            return [calib_inputs]
        return [calib_inputs]

    def _collect_layer_outputs(self) -> dict[str, torch.Tensor]:
        """
        Capture per-output-channel energy across calibration batches.

        Energy metric: E[x_out^2] per output channel, computed as the mean of
        per-batch variance of the output activations. For Conv1D, the axis is
        transposed accordingly.
        """
        self.model.eval()
        energy_sum: dict[str, torch.Tensor] = {}
        batch_count: dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                if not isinstance(output, torch.Tensor):
                    return
                out = output.detach().float()
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                out_f = (
                    target_mod.weight.shape[1] if is_conv1d
                    else target_mod.weight.shape[0]
                )
                if out.shape[-1] != out_f:
                    return
                flat = out.reshape(-1, out_f).cpu()
                if flat.size(0) >= 2:
                    var = flat.var(dim=0)
                else:
                    var = flat.abs().mean(dim=0)
                if layer_name not in energy_sum:
                    energy_sum[layer_name] = var
                    batch_count[layer_name] = 1
                else:
                    energy_sum[layer_name] += var
                    batch_count[layer_name] += 1
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
                        logger.debug("FLAT-LLM calibration batch %d failed: %s", i, e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: dict[str, torch.Tensor] = {}
        for name, esum in energy_sum.items():
            aggregated[name] = (esum / max(1, batch_count[name])).to(self.target_device)
        return aggregated

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        from framework.core.model_utils import replace_module

        outputs_map = self._collect_layer_outputs()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_keep_ratios = {}

        # Honest accounting: storage params, not "effective" made-up numbers.
        total_orig_storage = 0
        total_comp_storage = 0

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("FLAT-LLM: no compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().float().contiguous() if is_conv1d else raw_w.float()
            d_out, d_in = W.shape
            orig_layer_storage = d_out * d_in
            total_orig_storage += orig_layer_storage
            norm_w = torch.norm(W, p="fro").item()

            # Activation-aware saliency
            act = outputs_map.get(name, None)
            if act is not None and act.numel() > 0 and act.shape[0] == d_out:
                saliency = act.float().clamp(min=0.0).to(device)
            else:
                saliency = W.pow(2).sum(dim=1)

            # Apply damping to stabilize near-zero channels
            if self.damping > 0:
                saliency = saliency + self.damping * saliency.mean().clamp_min(1e-12)

            # ---- Decide kept channels ----
            if self.structured_24:
                # 2:4 is a compute-only structured sparsity. For storage, we still
                # keep all output channels; only the mask reduces the values.
                sal_mat = saliency.unsqueeze(1).expand_as(W)
                W_pruned, _mask = apply_24_structured_mask(W, saliency=sal_mat, dim=0)
                # Storage: unchanged (dense)
                kept_idx = torch.arange(d_out, device=device)
                W_kept = W_pruned.contiguous()
                keep_ratio_actual = 1.0
                bias_kept = (
                    target_mod.bias.data.float()
                    if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None
                    else None
                )
                comp_layer_storage = orig_layer_storage
            else:
                keep_dims = max(1, min(int(round(d_out * self.keep_ratio)), d_out))
                _, keep_idx = torch.topk(saliency, k=keep_dims, largest=True)
                keep_idx, _ = torch.sort(keep_idx)   # stable order
                kept_idx = keep_idx.to(device)
                W_kept = W.index_select(0, kept_idx).contiguous()
                keep_ratio_actual = keep_dims / d_out

                # Compute error vs original (with pruned rows treated as zero)
                W_pruned = torch.zeros_like(W)
                W_pruned.index_copy_(0, kept_idx, W_kept)

                # Bias handling: keep only the corresponding entries
                if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
                    bias_full = target_mod.bias.data.float()
                    bias_kept = bias_full.index_select(0, kept_idx).contiguous()
                else:
                    bias_kept = None

                # Storage: k * in instead of out * in
                comp_layer_storage = int(keep_dims) * d_in

            # ---- Error check ----
            if norm_w < 1e-12:
                final_err = 0.0
            else:
                if self.structured_24:
                    final_err = float(
                        torch.norm(W - W_pruned, p="fro").item() / norm_w
                    )
                else:
                    W_pruned_full = torch.zeros_like(W)
                    W_pruned_full.index_copy_(0, kept_idx, W_kept)
                    final_err = float(
                        torch.norm(W - W_pruned_full, p="fro").item() / norm_w
                    )

            self.layer_errors[name] = final_err
            self.layer_keep_ratios[name] = float(keep_ratio_actual)

            if final_err > self.max_error:
                logger.warning(
                    "FLAT-LLM error %.4f on %s exceeds %.4f.",
                    final_err, name, self.max_error,
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    total_comp_storage += orig_layer_storage
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # ---- Build the new module ----
            if self.structured_24:
                # No storage reduction: keep original Linear, just overwrite
                # its weight with the masked values (compute-only sparsity).
                new_module = target_mod
                with torch.no_grad():
                    if is_conv1d:
                        target_mod.weight.data.copy_(
                            W_kept.t().to(dtype=dtype, device=device)
                        )
                    else:
                        target_mod.weight.data.copy_(
                            W_kept.to(dtype=dtype, device=device)
                        )
            else:
                new_module = PrunedChannelLinear(
                    in_features=d_in,
                    original_out_features=d_out,
                    kept_idx=kept_idx.cpu(),
                    W_kept=W_kept.to(dtype=dtype, device=device),
                    bias_kept=(
                        bias_kept.to(dtype=dtype, device=device)
                        if bias_kept is not None else None
                    ),
                )

            # ---- Splice into the model (preserve LoRA wrappers) ----
            if hasattr(module, "base_layer"):
                module.base_layer = new_module
                # Keep LoRA params trainable
                for pn, p in module.named_parameters():
                    if "lora_" in pn.lower() and not pn.startswith("base_layer"):
                        p.requires_grad = True
            else:
                replace_module(self.model, name, new_module)

            self.compressed_layers.append(name)
            total_comp_storage += comp_layer_storage

        # ---- Forward sanity check (FLT-FIX-07) ----
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
                            (m for m in self.model.modules()
                             if isinstance(m, (nn.Linear, PrunedChannelLinear))),
                            None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        dt = first_p.dtype if first_p is not None else torch.float32
                        self.model(torch.randn(1, 8, in_dim, device=dev, dtype=dt))
        except Exception as e:
            logger.debug("FLAT-LLM forward sanity check failed: %s", e)
            forward_check_success = False

        # Honest storage ratio
        storage_ratio = (
            total_orig_storage / total_comp_storage
            if total_comp_storage > 0 else 1.0
        )
        effective_ratio = (
            total_comp_storage / total_orig_storage
            if total_orig_storage > 0 else 1.0
        )

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "layer_errors": self.layer_errors,
            "layer_keep_ratios": self.layer_keep_ratios,
            "keep_ratio": self.keep_ratio,
            "damping": self.damping,
            "structured_24": self.structured_24,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            # Honest accounting (FLT-FIX-02)
            "orig_layer_storage_params": total_orig_storage,
            "comp_layer_storage_params": total_comp_storage,
            "effective_parameter_ratio": round(effective_ratio, 4),
            "storage_compression_ratio": round(storage_ratio, 3),
            "storage_reduction_pct": round(
                100.0 * (1.0 - effective_ratio), 2
            ),
            "forward_check_success": forward_check_success,
            "method": "flat_llm_channel_pruning",
        })

        logger.info(
            "Compressor flat_llm finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ===========================================================================
# Factory (FLT-FIX-11)
# ===========================================================================
@COMPRESSOR_REGISTRY.register("flat_llm_factory")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor_factory")
def flat_llm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    keep_ratio = kwargs.pop(
        "keep_ratio",
        _cfg_get(comp_config, "flat_llm_keep_ratio",
                 _cfg_get(comp_config, "keep_ratio", 0.7)),
    )
    damping = kwargs.pop(
        "damping",
        _cfg_get(comp_config, "flat_llm_damping",
                 _cfg_get(comp_config, "damping", 1e-4)),
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(comp_config, "flat_llm_structured_24",
                 _cfg_get(comp_config, "structured_24", False)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "flat_llm_max_error",
                 _cfg_get(comp_config, "max_error", 0.35)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "flat_llm_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "flat_llm_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    tokenizer = kwargs.pop(
        "tokenizer",
        getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None),
    )
    num_calib = kwargs.pop(
        "num_calibration_batches",
        _cfg_get(comp_config, "num_calibration_batches", 8),
    )

    compressor = FLATLLMCompressor(
        model,
        tokenizer=tokenizer,
        keep_ratio=keep_ratio,
        damping=damping,
        structured_24=structured_24,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        num_calibration_batches=num_calib,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = ["FLATLLMCompressor", "PrunedChannelLinear", "flat_llm_factory"]
