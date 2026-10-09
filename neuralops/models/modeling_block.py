"""Transformer block module for NeuralOps models."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions  # Ensure attentions are registered
import neuralops.compression_engine  # Ensure compressors are registered

from neuralops.models.configuration import NeuralOpsConfig
from neuralops.models.modeling_norm import build_norm_layer
from neuralops.models.modeling_mlp import build_mlp


__all__ = ["NeuralOpsBlock"]


class NeuralOpsBlock(nn.Module):
    """A single Transformer Decoder layer supporting dynamic attention & compression."""

    def __init__(self, config: NeuralOpsConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention and pre-FFN normalization
        self.input_layernorm = build_norm_layer(
            config.norm_type, config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = build_norm_layer(
            config.norm_type, config.hidden_size, eps=config.rms_norm_eps
        )

        # Retrieve per-layer attention specification
        attn_spec = config.get_layer_attn_spec(layer_idx)
        attn_type = attn_spec["type"]
        if attn_type not in ATTENTION_REGISTRY:
            raise KeyError(
                f"Attention mechanism '{attn_type}' not found in ATTENTION_REGISTRY."
            )
        attn_cls = ATTENTION_REGISTRY[attn_type]

        # Extract extra arguments for attention mechanism
        attn_kwargs = {k: v for k, v in attn_spec.items() if k != "type"}
        try:
            self.self_attn = attn_cls(config, **attn_kwargs)
        except TypeError:
            try:
                self.self_attn = attn_cls(**attn_kwargs)
            except TypeError:
                self.self_attn = attn_cls()

        # MLP
        self.mlp = build_mlp(
            config.mlp_type,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # Apply compression layer if specified
        comp_spec = config.get_layer_compression_spec(layer_idx)
        if comp_spec is not None:
            comp_type = comp_spec["type"]
            if comp_type in COMPRESSOR_REGISTRY:
                comp_cls = COMPRESSOR_REGISTRY[comp_type]
                comp_kwargs = {k: v for k, v in comp_spec.items() if k != "type"}
                # Apply compressor to block or self_attn
                try:
                    compressor = comp_cls(self.self_attn, **comp_kwargs)
                    self.self_attn = compressor.compress()
                except Exception:
                    pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None, Any | None]:
        residual = hidden_states

        normed_hidden_states = self.input_layernorm(hidden_states)

        # Self Attention forward
        attn_outputs = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None
        else:
            attn_output = attn_outputs
            attn_weights = None
            present_key_value = None

        hidden_states = residual + attn_output

        # Fully Connected Feed-Forward
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
