#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FLA Layer Adapter for Hugging Face Transformers.

Wraps official Flash Linear Attention (FLA) layers to provide exact
Hugging Face transformer attention signature and return-arity compatibility.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.integrations.fla.cache_translator import (
    _translate_hf_cache_to_fla_state,
    _translate_fla_state_to_hf_cache,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_FAMILIES = (
    "llama",
    "mistral",
    "qwen2",
    "gpt2",
    "bert",
    "roberta",
)

CAUSAL_FAMILIES = (
    "llama",
    "mistral",
    "qwen2",
    "gpt2",
)


def _detect_family(config: Any, parent_family: Optional[str] = None) -> str:
    """Detect model family from explicit string or model config."""
    if parent_family is not None:
        return str(parent_family).lower()

    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type:
            mt = str(model_type).lower()
            if "llama" in mt:
                return "llama"
            if "mistral" in mt:
                return "mistral"
            if "qwen2" in mt or "qwen" in mt:
                return "qwen2"
            if "gpt2" in mt or "gpt" in mt:
                return "gpt2"
            if "roberta" in mt:
                return "roberta"
            if "bert" in mt:
                return "bert"

    return "unknown"


class FLALayerAdapter(nn.Module):
    """
    Adapter wrapping official FLA layers to provide strict Hugging Face
    transformer attention signature and return-arity compatibility.
    """

    def __init__(
        self,
        fla_layer_cls: Any,
        config: Any = None,
        parent_family: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        from neuralops.integrations.fla import fla_is_available

        if not fla_is_available():
            raise IntegrationNotImplementedError(
                "The 'fla' package is required to instantiate FLALayerAdapter. "
                "Install it via: pip install 'neuralops[fla]' (requires Linux + CUDA)."
            )

        fam = _detect_family(config, parent_family)
        if fam not in SUPPORTED_MODEL_FAMILIES:
            raise IntegrationNotImplementedError(
                f"Unsupported parent model family '{fam}' for FLA layer adapter. "
                f"Supported families are: {SUPPORTED_MODEL_FAMILIES}."
            )

        self.family = fam
        self.return_arity = 3 if fam in CAUSAL_FAMILIES else 2

        if fla_layer_cls is None:
            raise IntegrationNotImplementedError(
                "fla_layer_cls cannot be None when instantiating FLALayerAdapter."
            )

        # Deferred initialization of inner FLA layer
        hidden_size = getattr(config, "hidden_size", 768)
        num_heads = getattr(config, "num_attention_heads", 12)
        self.inner = fla_layer_cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs,
        )

        # Inspect if inner forward accepts attention_mask
        inner_forward_params = inspect.signature(self.inner.forward).parameters
        self.inner_accepts_mask = "attention_mask" in inner_forward_params or "mask" in inner_forward_params
        self.inner_mask_kwarg = "attention_mask" if "attention_mask" in inner_forward_params else "mask"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        """
        Forward pass providing Hugging Face signature compatibility.
        """
        fla_state = _translate_hf_cache_to_fla_state(past_key_value)

        forward_kwargs: dict[str, Any] = dict(kwargs)
        if use_cache:
            forward_kwargs["use_cache"] = True
        if output_attentions:
            forward_kwargs["output_attentions"] = True

        if fla_state is not None:
            forward_kwargs["past_key_values"] = fla_state

        if attention_mask is not None and self.inner_accepts_mask:
            forward_kwargs[self.inner_mask_kwarg] = attention_mask

        # Execute inner FLA forward pass
        raw_output = self.inner(hidden_states, **forward_kwargs)

        if isinstance(raw_output, tuple):
            attn_output = raw_output[0]
            next_state = raw_output[1] if len(raw_output) > 1 else None
        else:
            attn_output = raw_output
            next_state = None

        # Post-hoc mask application if inner layer does not accept mask
        if attention_mask is not None and not self.inner_accepts_mask:
            if attention_mask.dim() == 2:
                attn_output = attn_output * attention_mask.unsqueeze(-1).to(attn_output.dtype)
            elif attention_mask.dim() == 4:
                # 4D mask: zero out masked positions
                mask_2d = (attention_mask[:, :, -1, :] > -1e-4).float().mean(dim=1)
                attn_output = attn_output * mask_2d.unsqueeze(-1).to(attn_output.dtype)

        translated_state = _translate_fla_state_to_hf_cache(next_state)

        # Arity negotiation
        if self.return_arity == 3:
            # LLaMA / GPT2 family expects (attn_output, attn_weights, past_key_value)
            return (attn_output, None, translated_state)
        # BERT / RoBERTa family expects (context_layer, attention_probs)
        return (attn_output, None)
