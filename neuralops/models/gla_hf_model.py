"""
HuggingFace Pretrained Configuration and CausalLM Model for GLA.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from neuralops.attentions.linear.gla_attention import GLAttention


class GLAConfig(PretrainedConfig):
    model_type = "gla"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        gate_rank: int = 16,
        gate_tau: float = 16.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.gate_rank = gate_rank
        self.gate_tau = gate_tau


class GLABlock(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = GLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            gate_rank=config.gate_rank,
            gate_tau=config.gate_tau,
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out, _ = self.attn(hidden_states)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class GLAModel(PreTrainedModel, GenerationMixin):
    config_class = GLAConfig
    base_model_prefix = "gla"
    main_input_name = "input_ids"

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GLABlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": False,
            **kwargs,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        B, N = input_ids.shape
        pos = torch.arange(0, N, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(pos)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
