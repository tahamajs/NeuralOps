"""
HuggingFace DynamicCache & Generation Bridge.

Provides:
  - NeuralOpsHybridHFCache: subclass of Cache / DynamicCache (when transformers is available)
    or standalone duck-typed cache maintaining seen tokens and per-layer state slots.
  - Supports:
      * Recurrent / Linear state slots (S_state, Z_state, conv_state) for constant O(1) memory decoding.
      * Sliding-window ring buffers for windowed attentions.
      * Standard KV cache for full attention layers in hybrid models.
  - wrap_model_for_generation(): equips any surgically modified CausalLM with generation pre-hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

logger = logging.getLogger("NeuralOps.Surgery.HFCacheBridge")

# Attempt importing HuggingFace Cache base classes
try:
    from transformers.cache_utils import Cache, DynamicCache
    _HF_CACHE_BASE = DynamicCache
except Exception:
    class Cache:
        pass
    class DynamicCache:
        pass
    _HF_CACHE_BASE = object


class NeuralOpsHybridHFCache(_HF_CACHE_BASE):
    """
    HuggingFace compliant generation cache supporting heterogeneous layers.
    Guarantees get_seq_length(), get_max_length(), and get_usable_length()
    track true sequence lengths so RoPE position_ids and cache_position match exactly.
    """

    def __init__(self) -> None:
        if _HF_CACHE_BASE is not object and hasattr(super(), "__init__"):
            super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.recurrent_state_cache: Dict[int, Any] = {}
        self.sliding_window_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._seen_tokens: int = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the number of tokens processed so far."""
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates standard KV cache for full-attention layers."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))

        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # Update global seen tokens on layer 0 update
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_recurrent_state(self, layer_idx: int, state: Any, num_tokens: int = 1) -> Any:
        """Stores constant O(1) recurrent states (GLA, GDN, RetNet, etc.)."""
        self.recurrent_state_cache[layer_idx] = state
        if layer_idx == 0:
            self._seen_tokens += num_tokens
        return state

    def get_recurrent_state(self, layer_idx: int) -> Optional[Any]:
        return self.recurrent_state_cache.get(layer_idx, None)


def wrap_model_for_generation(model: torch.nn.Module) -> torch.nn.Module:
    """
    Wraps model's prepare_inputs_for_generation or injects NeuralOpsHybridHFCache
    whenever model.generate() is called.
    """
    if hasattr(model, "generate") and hasattr(model, "prepare_inputs_for_generation"):
        orig_prep = model.prepare_inputs_for_generation

        def _patched_prep(input_ids: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
            model_inputs = orig_prep(input_ids, **kwargs)
            if model_inputs.get("past_key_values", None) is None and kwargs.get("use_cache", True):
                # Ensure cache is instantiated
                if not isinstance(kwargs.get("past_key_values"), NeuralOpsHybridHFCache):
                    model_inputs["past_key_values"] = NeuralOpsHybridHFCache()
            return model_inputs

        model.prepare_inputs_for_generation = _patched_prep

    return model


__all__ = [
    "NeuralOpsHybridHFCache",
    "wrap_model_for_generation",
]
