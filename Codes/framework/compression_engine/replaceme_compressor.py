"""
ReplaceMe: Layer Removal and Replacement with Linear Residual Block.

Replaces the last N layers of a Transformer encoder or decoder with a compatible
LayerNorm + Linear residual block with optional trainable LoRA fine-tuning adapters.

Registered in COMPRESSOR_REGISTRY as 'replaceme', 'replaceme_compressor', and 'replaceme_factory'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# ReplacementBlock
# ===========================================================================
class ReplacementBlock(nn.Module):
    """
    Drop-in replacement for a transformer layer.

    Preserves tensor shapes and maintains signal propagation via a residual
    connection with optional LoRA fine-tuning adapters.
    """
    def __init__(
        self,
        hidden_size: int,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        return_tuple: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.return_tuple = bool(return_tuple)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        self.ln = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        # Zero-init → block starts as near-identity
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        if self.lora_r > 0:
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, hidden_size), dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((hidden_size, self.lora_r), dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | Any,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Extract primary tensor
        if isinstance(hidden_states, (tuple, list)):
            primary = hidden_states[0]
        elif hasattr(hidden_states, "last_hidden_state"):
            primary = hidden_states.last_hidden_state
        else:
            primary = hidden_states

        normed = self.ln(primary)
        delta = self.linear(normed)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_delta = F.linear(F.linear(normed.float(), self.lora_A), self.lora_B) * self.scaling
            delta = delta + lora_delta.to(delta.dtype)

        out = primary + delta

        # RM-FIX-02: return correct tuple shape
        if self.return_tuple:
            return (out,)
        return out


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("replaceme")
@COMPRESSOR_REGISTRY.register("replaceme_compressor")
class ReplaceMeCompressor(BaseCompressor):
    """
    Replaces the trailing N layers of a Transformer with a lightweight
    LayerNorm + Linear residual block.
    """

    def __init__(
        self,
        model: nn.Module,
        remove_layers: int = 2,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        fallback_to_original: bool = True,
        device: str | torch.device | None = None,
        config: Any | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="replaceme", **kwargs)
        cfg_obj = cfg or config or kwargs.get("comp_config")

        self.remove_layers = max(
            1, int(remove_layers if remove_layers is not None
                   else _cfg_get(cfg_obj, "remove_layers", 2))
        )
        self.use_lora = bool(
            use_lora
            or _cfg_get(cfg_obj, "use_lora", False)
            or (lora_r > 0)
        )
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.fallback_to_original = bool(
            fallback_to_original
            if fallback_to_original is not None
            else _cfg_get(cfg_obj, "fallback_to_original", True)
        )

        # Device resolution
        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = (
                    next(model.parameters()).device
                    if list(model.parameters())
                    else torch.device("cpu")
                )
        else:
            self.target_device = (
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )

    # ------------------------------------------------------------------
    def _find_transformer_layers(self) -> nn.ModuleList | None:
        """Locate the transformer layer list across common architectures."""
        # Ordered list of attribute paths (most specific first).
        candidates = [
            ("distilbert", "transformer", "layer"),
            ("roberta", "encoder", "layer"),
            ("bert", "encoder", "layer"),
            ("encoder", "layer"),
            ("transformer", "h"),      # GPT-2
            ("transformer", "layer"),
            ("model", "layers"),       # LLaMA / Qwen
        ]
        for path in candidates:
            curr = self.model
            ok = True
            for attr in path:
                if hasattr(curr, attr):
                    curr = getattr(curr, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(curr, (nn.ModuleList, list)) and len(curr) > 0:
                return curr

        # Fallback: scan for the first long ModuleList whose elements look like
        # transformer layers.
        for _, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 2:
                first_cls = module[0].__class__.__name__.lower()
                if any(k in first_cls for k in ("layer", "block", "decoderlayer", "encoderlayer")):
                    return module
        return None

    # ------------------------------------------------------------------
    def _infer_hidden_size(self, layers: nn.ModuleList) -> int:
        # Try config first.
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for key in ("hidden_size", "d_model", "dim", "n_embd"):
                v = _cfg_get(cfg, key, None)
                if isinstance(v, int) and v > 0:
                    return int(v)
        # Fallback: inspect first layer's parameters.
        try:
            first_layer = layers[0]
            for p in first_layer.parameters():
                if p.dim() == 2:
                    return int(p.shape[-1])
        except Exception:
            pass
        return 768  # last-resort default

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        layers = self._find_transformer_layers()
        layers_replaced = 0
        fallback_layers: list[int] = []

        if layers is None:
            logger.warning("ReplaceMe: could not locate transformer layer list.")
            stats = self.get_compression_stats()
            stats.update({
                "layers_removed": 0,
                "layers_replaced": 0,
                "fallback_layers": [],
                "fallback_count": 0,
                "target_removed": self.remove_layers,
                "hidden_size": None,
                "forward_check_success": False,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "method": "replaceme",
            })
            self.stats = stats
            return self.model, stats

        hidden_size = self._infer_hidden_size(layers)
        num_layers = len(layers)
        n_remove = min(self.remove_layers, num_layers)

        target_dtype = (
            next(self.model.parameters()).dtype
            if list(self.model.parameters())
            else torch.float32
        )

        for i in range(num_layers - n_remove, num_layers):
            old_layer = layers[i]

            # Detect whether the original layer returns a tuple.
            try:
                dummy_h = torch.randn(
                    1, 4, hidden_size,
                    device=self.target_device, dtype=target_dtype,
                )
                with torch.no_grad():
                    old_out = old_layer(dummy_h)
                return_tuple = isinstance(old_out, (tuple, list))
            except Exception as e:
                logger.debug(
                    "ReplaceMe: could not probe layer %d output type (%s); "
                    "defaulting to tuple=True.", i, e,
                )
                return_tuple = True

            new_block = ReplacementBlock(
                hidden_size,
                lora_r=self.lora_r if self.use_lora else 0,
                lora_alpha=self.lora_alpha,
                return_tuple=return_tuple,
                device=self.target_device,
                dtype=target_dtype,
            )
            layers[i] = new_block
            layers_replaced += 1

        # Forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy_input = torch.randint(
                    0, 100, (1, 8), device=self.target_device,
                )
                try:
                    self.model(input_ids=dummy_input)
                except Exception:
                    # Some non-HF models expect positional arg.
                    self.model(dummy_input)
        except Exception as e:
            logger.warning("ReplaceMe forward sanity check failed: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "layers_removed": layers_replaced,
            "layers_replaced": layers_replaced,
            "fallback_layers": fallback_layers,   # RM-FIX-03: list of indices
            "fallback_count": len(fallback_layers),
            "target_removed": self.remove_layers,
            "hidden_size": hidden_size,
            "forward_check_success": forward_check_success,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "method": "replaceme",
        })
        self.stats = stats
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("replaceme_factory")
@COMPRESSOR_REGISTRY.register("replaceme_compressor_factory")
def replaceme_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """ReplaceMe compressor factory function."""
    remove_layers = kwargs.pop(
        "remove_layers", _cfg_get(comp_config, "remove_layers", 2)
    )
    use_lora = kwargs.pop(
        "use_lora", _cfg_get(comp_config, "use_lora", False)
    )
    lora_r = kwargs.pop(
        "lora_r", _cfg_get(comp_config, "lora_r", 0)
    )
    lora_alpha = kwargs.pop(
        "lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0)
    )
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )

    compressor = ReplaceMeCompressor(
        model,
        remove_layers=remove_layers,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


replaceme_compressor_factory = replaceme_factory

__all__ = [
    "ReplacementBlock",
    "ReplaceMeCompressor",
    "replaceme_factory",
    "replaceme_compressor_factory",
]
