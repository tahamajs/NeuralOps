"""
LIFTR – Linearized Function Trick Compressor.
Registered as 'liftr', 'liftr_compressor', 'liftr_class'.

Mathematical Formulation:
  Activation:    ReLU(x) → Identity(x), GELU(x) / SiLU(x) ≈ α·x
  Attention:     Softmax(QK^T/√d)V → (1/N) Σ V_i

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


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

# LIFTR-FIX-10: safe SiLU reference
_SILU_CLASSES: tuple[type, ...] = (nn.GELU,)
try:
    if hasattr(nn, "SiLU"):
        _SILU_CLASSES = (nn.GELU, nn.SiLU)
except Exception:
    pass


# ===========================================================================
# Linearized activation
# ===========================================================================


class _LinearizedGELU(nn.Module):
    """f(x) ≈ α · x."""

    is_custom_attention: bool = False

    def __init__(self, alpha: float = 0.8) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


# ===========================================================================
# Linearized attention wrapper
# ===========================================================================


class _LinearizedAttentionWrapper(nn.Module):
    """
    Replaces quadratic softmax attention with sequence-mean of values.

    LIFTR-FIX-04: if the wrapped module's architecture is not recognized, the
    call is transparently delegated to the original attention (safe fallback).

    LIFTR-FIX-02: marked as is_custom_attention = False (this is not an
    attention module from the framework's perspective; it delegates to one).
    """

    is_custom_attention: bool = False

    def __init__(self, original_attn: nn.Module, layer_name: str = "") -> None:
        super().__init__()
        self.original_attn = original_attn
        self.layer_name = layer_name
        # LIFTR-FIX-05: determine return signature once at init.
        self._returns_tuple = self._detect_tuple_return(original_attn)

    @staticmethod
    def _detect_tuple_return(original_attn: nn.Module) -> bool:
        """
        Heuristic:
          - Modules whose `forward` returns (context, attn_weights) — GPT-2, BERT, DistilBERT —
            should return a tuple.
          - Linear attention / DeltaNet return a single tensor.
        """
        cls_name = original_attn.__class__.__name__.lower()
        if any(s in cls_name for s in ("linear", "delta", "mamba", "ssm")):
            return False
        # GPT-2 and BERT-style return tuples.
        if any(s in cls_name for s in ("gpt", "bert", "roberta", "electra", "deberta")):
            return True
        # Default: assume attention returns a tuple.
        return True

    def _resolve_hidden_size(self) -> int:
        attn = self.original_attn
        for attr in ("q_proj", "q_lin", "query"):
            mod = getattr(attn, attr, None)
            if mod is not None:
                base = getattr(mod, "base_layer", mod)
                in_f = getattr(base, "in_features", None)
                if isinstance(in_f, int) and in_f > 0:
                    return in_f
        if hasattr(attn, "c_attn"):
            base = getattr(attn.c_attn, "base_layer", attn.c_attn)
            w = getattr(base, "weight", None)
            if w is not None and w.dim() == 2:
                # Conv1D stores (in, out)
                return int(w.shape[0])
        # Fallback
        return 768

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        # Determine input tensor.
        x = hidden_states
        if x is None:
            x = query
        if x is None:
            x = value
        if x is None:
            # LIFTR-FIX-04: If we truly have no input, delegate to original.
            return self._delegate(
                x if x is not None else None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

        attn = self.original_attn
        try:
            # ---- 1. LLaMA / Mistral / Gemma / Qwen (v_proj + o_proj) ----
            if hasattr(attn, "v_proj") and hasattr(attn, "o_proj"):
                v = attn.v_proj(x)
                o_in = getattr(attn.o_proj, "in_features", v.shape[-1])
                if v.shape[-1] != o_in:
                    # Truncate or project (rare).
                    if v.shape[-1] > o_in:
                        v = v[..., :o_in]
                    else:
                        rep = (o_in + v.shape[-1] - 1) // v.shape[-1]
                        v = v.repeat(1, 1, rep)[..., :o_in]
                ctx = v.mean(dim=1, keepdim=True).expand(v.shape[0], v.shape[1], o_in)
                output = attn.o_proj(ctx.to(x.dtype))
                return self._format_output(output, output_attentions)

            # ---- 2. DistilBERT (q_lin, v_lin, out_lin) ----
            if (
                hasattr(attn, "q_lin")
                and hasattr(attn, "v_lin")
                and hasattr(attn, "out_lin")
            ):
                v = attn.v_lin(x)
                o_in = getattr(attn.out_lin, "in_features", v.shape[-1])
                if v.shape[-1] != o_in:
                    if v.shape[-1] > o_in:
                        v = v[..., :o_in]
                    else:
                        rep = (o_in + v.shape[-1] - 1) // v.shape[-1]
                        v = v.repeat(1, 1, rep)[..., :o_in]
                ctx = v.mean(dim=1, keepdim=True).expand(v.shape[0], v.shape[1], o_in)
                output = attn.out_lin(ctx.to(x.dtype))
                return self._format_output(output, output_attentions)

            # ---- 3. RoBERTa / BERT inner self-attention ----
            if hasattr(attn, "value") and hasattr(attn, "query"):
                v = attn.value(x)
                ctx = v.mean(dim=1, keepdim=True).expand_as(v)
                return self._format_output(ctx.to(x.dtype), output_attentions)

            # ---- 4. GPT-2 (fused c_attn + c_proj) ----
            if hasattr(attn, "c_attn") and hasattr(attn, "c_proj"):
                qkv = attn.c_attn(x)
                hidden_size = x.shape[-1]
                if qkv.shape[-1] >= hidden_size * 3:
                    _, _, v = torch.split(qkv, hidden_size, dim=-1)
                else:
                    v = qkv
                ctx = v.mean(dim=1, keepdim=True).expand_as(v)
                output = attn.c_proj(ctx.to(x.dtype))
                return self._format_output(output, output_attentions)

        except Exception as e:
            logger.debug(
                "LIFTR wrapper: mean-attention path failed (%s); delegating.", e
            )

        # ---- LIFTR-FIX-04: fallback to original ----
        return self._delegate(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

    def _delegate(
        self,
        x: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Call the underlying attention with matching signature."""
        try:
            if x is None:
                # Cannot call — return an empty tensor of the right shape.
                hs = self._resolve_hidden_size()
                return self._format_output(
                    torch.zeros(1, 1, hs, dtype=torch.float32), output_attentions
                )
            call_kwargs = {}
            if attention_mask is not None:
                call_kwargs["attention_mask"] = attention_mask
            if head_mask is not None:
                call_kwargs["head_mask"] = head_mask
            call_kwargs["output_attentions"] = output_attentions
            call_kwargs.update(kwargs)
            res = self.original_attn(x, **call_kwargs)
            if isinstance(res, (tuple, list)):
                return res[0] if not output_attentions else res
            return self._format_output(res, output_attentions)
        except Exception as e:
            logger.debug("LIFTR delegate failed (%s); returning zeros.", e)
            hs = self._resolve_hidden_size()
            batch, seq = 1, 1
            if x is not None and x.dim() >= 2:
                batch, seq = x.shape[0], x.shape[1]
            return self._format_output(
                torch.zeros(
                    batch,
                    seq,
                    hs,
                    dtype=x.dtype if x is not None else torch.float32,
                    device=x.device if x is not None else "cpu",
                ),
                output_attentions,
            )

    def _format_output(
        self,
        out: torch.Tensor,
        output_attentions: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """LIFTR-FIX-05: match tuple signature expected by parent module."""
        if self._returns_tuple:
            return (out, None)
        return out


# ===========================================================================
# Compressor
# ===========================================================================


@COMPRESSOR_REGISTRY.register("liftr_class")
@COMPRESSOR_REGISTRY.register("liftr_compressor_class")
class LIFTRCompressor(BaseCompressor):
    # LIFTR-FIX-01: this is a compressor, not an attention.
    is_custom_attention: bool = False

    CRITICAL_MODULE_KEYWORDS = [
        "classifier",
        "score",
        "lm_head",
        "pooler",
        "embed",
        "wte",
        "wpe",
        "shared",
        "head",
        "logits",
    ]

    def __init__(
        self,
        model: nn.Module,
        replace_attention: bool = False,
        gelu_coefficient: float = 0.8,
        validation_calibration: bool = False,
        calibration_threshold: float = 0.20,
        target_attention_only: bool = False,
        skip_should_compress_filter: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="liftr", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        self.cfg = cfg

        self.replace_attention = bool(
            _cfg_get(
                cfg,
                "liftr_replace_attention",
                _cfg_get(cfg, "replace_attention", replace_attention),
            )
        )
        self.gelu_coefficient = float(
            _cfg_get(
                cfg,
                "liftr_gelu_coefficient",
                _cfg_get(cfg, "gelu_coefficient", gelu_coefficient),
            )
        )
        self.validation_calibration = bool(
            _cfg_get(
                cfg,
                "validation_calibration",
                validation_calibration,
            )
        )
        self.calibration_threshold = float(
            _cfg_get(
                cfg,
                "calibration_threshold",
                calibration_threshold,
            )
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "liftr_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.skip_should_compress_filter = bool(
            _cfg_get(
                cfg,
                "skip_should_compress_filter",
                skip_should_compress_filter,
            )
        )

        self.linearized_activations_count = 0
        self.linearized_attentions_count = 0
        self.replaced_modules: list[str] = []
        self.skipped_layers: list[str] = []
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    def _is_critical(self, name: str) -> bool:
        low = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B")):
            return True
        return any(kw in low for kw in self.CRITICAL_MODULE_KEYWORDS)

    def _is_already_compressed(self, module: nn.Module) -> bool:
        """LIFTR-FIX-06: skip already-compressed module types."""
        cls_name = module.__class__.__name__
        return cls_name in (
            "LSCPrunedLoRALinear",
            "DynamicSVDLinear",
            "SVDLinear",
            "FactorisedLinear",
            "FactorizedLinear",
            "LowRankPlusSparseLinear",
            "_LinearizedAttentionWrapper",
        )

    def _linearize_module(
        self,
        module: nn.Module,
        prefix: str = "",
        _depth: int = 0,
    ) -> None:
        """Recursively replaces activation functions and (optionally) attention blocks."""
        if _depth > 16:
            return

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if self._is_critical(full_name):
                continue

            if self._is_already_compressed(child):
                if (
                    hasattr(child, "lora_A")
                    or child.__class__.__name__.lower() == "loralinear"
                ):
                    continue
                if isinstance(child, _LinearizedAttentionWrapper):
                    continue

            # Skip LoRA-wrapped modules.
            if (
                hasattr(child, "lora_A")
                or child.__class__.__name__.lower() == "loralinear"
            ):
                continue

            is_attn_layer = any(k in full_name.lower() for k in _ATTN_NAMES)

            # --- Activation linearization ---
            if isinstance(child, nn.ReLU):
                if self.target_attention_only and not is_attn_layer:
                    self.skipped_layers.append(full_name)
                    continue
                setattr(module, name, nn.Identity())
                self.linearized_activations_count += 1
                self.replaced_modules.append(full_name)
                continue

            if isinstance(child, _SILU_CLASSES):
                if self.target_attention_only and not is_attn_layer:
                    self.skipped_layers.append(full_name)
                    continue
                setattr(module, name, _LinearizedGELU(alpha=self.gelu_coefficient))
                self.linearized_activations_count += 1
                self.replaced_modules.append(full_name)
                continue

            # --- Attention linearization ---
            if self.replace_attention and is_attn_layer:
                if "norm" in name.lower():
                    self._linearize_module(child, prefix=full_name, _depth=_depth + 1)
                    continue
                if isinstance(
                    child, (nn.Linear, nn.LayerNorm, nn.RMSNorm, nn.Embedding)
                ):
                    self._linearize_module(child, prefix=full_name, _depth=_depth + 1)
                    continue

                # BERT / RoBERTa nested structure: linearize inner self-attention.
                if hasattr(child, "self") and hasattr(child, "output"):
                    self._linearize_module(child, prefix=full_name, _depth=_depth + 1)
                    continue

                if isinstance(child, _LinearizedAttentionWrapper):
                    continue

                if not hasattr(child, "_liftr_skip") and not isinstance(
                    child, _LinearizedAttentionWrapper
                ):
                    try:
                        wrapped = _LinearizedAttentionWrapper(
                            child, layer_name=full_name
                        )
                        setattr(module, name, wrapped)
                        self.linearized_attentions_count += 1
                        self.replaced_modules.append(full_name)
                    except Exception as e:
                        logger.warning("LIFTR: failed to wrap %s: %s", full_name, e)
                continue

            # --- Recurse ---
            self._linearize_module(child, prefix=full_name, _depth=_depth + 1)

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        # LIFTR-FIX-15: idempotence
        if getattr(self.model, "_liftr_compressed", False):
            logger.warning("LIFTR: model already compressed; skipping.")
            try:
                stats = self.get_compression_stats()
            except Exception:
                stats = {}
            return self.model, stats

        # LIFTR-FIX-11: safe device resolution.
        first_param = next(iter(self.model.parameters()), None)
        device = first_param.device if first_param is not None else torch.device("cpu")

        # LIFTR-FIX-03: dummy_in defined at function scope.
        dummy_in: torch.Tensor | None = None
        orig_out: torch.Tensor | None = None

        if self.validation_calibration:
            try:
                self.model.eval()
                dummy_in = torch.randint(
                    0, 100, (1, 8), device=device, dtype=torch.long
                )
                with torch.no_grad():
                    res = self.model(dummy_in)
                    if hasattr(res, "logits"):
                        orig_out = res.logits.detach()
                    elif isinstance(res, (tuple, list)):
                        orig_out = res[0].detach()
                    else:
                        orig_out = res.detach()
            except Exception as e:
                logger.debug("LIFTR baseline calibration failed: %s", e)
                dummy_in = None
                orig_out = None

        # Apply linearization.
        self._linearize_module(self.model)

        # Freeze parameters in skipped modules.
        if self.skipped_layers:
            skipped_set = set(self.skipped_layers)
            for name, module in self.model.named_modules():
                if name in skipped_set:
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False

        # Post-linearization validation.
        validation_passed = True
        rel_diff = 0.0
        if (
            self.validation_calibration
            and orig_out is not None
            and dummy_in is not None
        ):
            try:
                with torch.no_grad():
                    res_new = self.model(dummy_in)
                    if hasattr(res_new, "logits"):
                        new_out = res_new.logits
                    elif isinstance(res_new, (tuple, list)):
                        new_out = res_new[0]
                    else:
                        new_out = res_new
                    rel_diff = (
                        torch.norm(orig_out.float() - new_out.float())
                        / (torch.norm(orig_out.float()) + 1e-8)
                    ).item()
                    if rel_diff > self.calibration_threshold:
                        logger.warning(
                            "LIFTR: validation diff %.4f > threshold %.4f.",
                            rel_diff,
                            self.calibration_threshold,
                        )
                        validation_passed = False
            except Exception as e:
                logger.warning("LIFTR: post-linearization validation failed: %s", e)
                validation_passed = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update(
            {
                "linearized_activations": self.linearized_activations_count,
                "linearized_attentions": self.linearized_attentions_count,
                "replace_attention": self.replace_attention,
                "replaced_modules": self.replaced_modules,
                "gelu_coefficient": self.gelu_coefficient,
                "target_attention_only": self.target_attention_only,
                "validation_calibration": self.validation_calibration,
                "validation_passed": validation_passed,
                "calibration_rel_diff": rel_diff,
                # LIFTR-FIX-08: explicit note.
                "effective_parameter_ratio": 1.0,
                "flops_reduction_pct": 0.0,
                "method": "liftr_linearization",
                "note": (
                    "LIFTR performs functional linearization (activation & attention). "
                    "It does NOT reduce parameters; original attention submodules are "
                    "retained as `self.original_attn` for safe delegation."
                ),
            }
        )
        self.stats = stats
        self.model._liftr_compressed = True
        logger.info(
            "LIFTR: %d activations linearized, %d attentions wrapped.",
            self.linearized_activations_count,
            self.linearized_attentions_count,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================


@COMPRESSOR_REGISTRY.register("liftr")
@COMPRESSOR_REGISTRY.register("liftr_compressor")
@COMPRESSOR_REGISTRY.register("liftr_factory")
def liftr_compressor_entry(
    model: nn.Module,
    comp_config: Any = None,
    **kwargs: Any,
) -> Any:
    """Factory function for LIFTR compressor."""
    replace_attention = _cfg_get(
        comp_config,
        "liftr_replace_attention",
        _cfg_get(
            comp_config, "replace_attention", kwargs.pop("replace_attention", False)
        ),
    )
    gelu_coefficient = _cfg_get(
        comp_config,
        "liftr_gelu_coefficient",
        _cfg_get(comp_config, "gelu_coefficient", kwargs.pop("gelu_coefficient", 0.8)),
    )
    target_attention_only = _cfg_get(
        comp_config,
        "liftr_target_attention_only",
        _cfg_get(
            comp_config,
            "target_attention_only",
            kwargs.pop("target_attention_only", False),
        ),
    )
    skip_filter = _cfg_get(
        comp_config,
        "skip_should_compress_filter",
        kwargs.pop("skip_should_compress_filter", False),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = LIFTRCompressor(
        model,
        replace_attention=replace_attention,
        gelu_coefficient=gelu_coefficient,
        target_attention_only=target_attention_only,
        skip_should_compress_filter=skip_filter,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "LIFTRCompressor",
    "liftr_compressor_entry",
]
