"""
ReplaceMe: Layer Removal and Replacement with Linear Residual Block.
Replaces the last N layers of a Transformer encoder or decoder with a compatible
LayerNorm + Linear residual block with optional trainable LoRA fine-tuning adapters.
Registered in COMPRESSOR_REGISTRY as 'replaceme', 'replaceme_compressor', and 'replaceme_factory'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class ReplacementBlock(nn.Module):
    """
    Drop-in replacement for a transformer layer.
    Preserves tensor shapes, handles variable signature inputs (*args, **kwargs),
    and maintains signal propagation via a residual connection with optional LoRA parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        return_tuple: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.return_tuple = return_tuple
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        self.ln = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        # Initialize to zero so block acts as near-identity at initialization
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        if self.lora_r > 0:
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, hidden_size), dtype=torch.float32, device=device))
            self.lora_B = nn.Parameter(torch.zeros((hidden_size, self.lora_r), dtype=torch.float32, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        normed = self.ln(hidden_states)
        delta = self.linear(normed)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_delta = F.linear(F.linear(normed.float(), self.lora_A), self.lora_B) * self.scaling
            delta = delta + lora_delta.to(delta.dtype)
        out = hidden_states + delta
        if self.return_tuple:
            return (out,)
        return out


@COMPRESSOR_REGISTRY.register('replaceme')
@COMPRESSOR_REGISTRY.register('replaceme_compressor')
class ReplaceMeCompressor(BaseCompressor):
    """
    Replaces the trailing N layers of a Transformer with a lightweight LayerNorm + Linear residual block.
    """

    def __init__(
        self,
        model: nn.Module,
        remove_layers: int = 2,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[Any] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name='replaceme', **kwargs)
        cfg_obj = cfg or config or kwargs.get('comp_config', None)
        if cfg_obj is not None:
            remove_layers = getattr(cfg_obj, 'remove_layers', remove_layers)
            use_lora = getattr(cfg_obj, 'use_lora', use_lora)
            lora_r = getattr(cfg_obj, 'lora_r', lora_r)
            lora_alpha = getattr(cfg_obj, 'lora_alpha', lora_alpha)

        self.remove_layers = max(1, int(remove_layers))
        self.use_lora = bool(use_lora or (lora_r > 0))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)

        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        else:
            self.target_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

    def _find_transformer_layers(self) -> Optional[nn.ModuleList]:
        """Finds transformer layers across diverse architectures."""
        candidates = [
            ('encoder', 'layer'),
            ('roberta', 'encoder', 'layer'),
            ('bert', 'encoder', 'layer'),
            ('distilbert', 'transformer', 'layer'),
            ('transformer', 'layer'),
            ('transformer', 'h'),
            ('model', 'layers'),
            ('gpt2', 'transformer', 'h'),
        ]
        for path in candidates:
            curr = self.model
            found = True
            for attr in path:
                if hasattr(curr, attr):
                    curr = getattr(curr, attr)
                else:
                    found = False
                    break
            if found and isinstance(curr, (nn.ModuleList, list)):
                return curr

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 2:
                first = module[0]
                cls_lower = first.__class__.__name__.lower()
                if any(k in cls_lower for k in ['layer', 'block', 'decoderlayer', 'encoderlayer']):
                    return module
        return None

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        orig_params = self._count_params(self.model)

        # Resolve hidden_size
        hidden_size = 768
        cfg = getattr(self.model, 'config', None)
        if cfg is not None:
            hidden_size = getattr(cfg, 'hidden_size', getattr(cfg, 'dim', getattr(cfg, 'd_model', 768)))

        layers = self._find_transformer_layers()
        layers_replaced = 0

        target_dtype = next(self.model.parameters()).dtype if list(self.model.parameters()) else torch.float32

        if layers is not None:
            num_layers = len(layers)
            n_remove = min(self.remove_layers, num_layers)
            return_tuple = False
            for i in range(num_layers - n_remove, num_layers):
                layers[i] = ReplacementBlock(
                    hidden_size,
                    lora_r=self.lora_r if self.use_lora else 0,
                    lora_alpha=self.lora_alpha,
                    return_tuple=return_tuple,
                    device=self.target_device,
                    dtype=target_dtype,
                )
                layers_replaced += 1
        else:
            logger.warning('ReplaceMe: Could not locate transformer layer list to replace.')

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy_input = torch.randint(0, 100, (1, 8), device=self.target_device)
                self.model(dummy_input)
        except Exception as ex_sanity:
            logger.warning('ReplaceMe post-compression forward sanity check warning: %s', ex_sanity)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            'layers_removed': layers_replaced,
            'target_removed': self.remove_layers,
            'hidden_size': hidden_size,
            'forward_check_success': forward_check_success,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'method': 'replaceme',
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register('replaceme_factory')
@COMPRESSOR_REGISTRY.register('replaceme_compressor_factory')
def replaceme_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """ReplaceMe compressor factory function returning (compressed_model, stats) when requested."""
    remove_layers = kwargs.pop('remove_layers', getattr(comp_config, 'remove_layers', 2) if comp_config else 2)
    use_lora = kwargs.pop('use_lora', getattr(comp_config, 'use_lora', False) if comp_config else False)
    lora_r = kwargs.pop('lora_r', getattr(comp_config, 'lora_r', 0) if comp_config else 0)
    lora_alpha = kwargs.pop('lora_alpha', getattr(comp_config, 'lora_alpha', 1.0) if comp_config else 1.0)

    compressor = ReplaceMeCompressor(
        model,
        remove_layers=remove_layers,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop('return_both', False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    'ReplacementBlock',
    'ReplaceMeCompressor',
    'replaceme_factory',
]
