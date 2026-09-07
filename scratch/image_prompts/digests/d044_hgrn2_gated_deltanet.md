# d044 hgrn2_gated_deltanet  [attention]
aliases: hgrn2_gated_deltanet
file: Codes/framework/attentions/hgrn2_gated_deltanet.py  doc: Docs/content/attentions/hgrn2_gated_deltanet.md
doc: HGRN-2 Gated DeltaNet Hybrid Attention Module. | Combines bidirectional/causal HGRN-2 element-wise linear recurrence with multi-head | Gated DeltaNet associative matrix memory for rich representational capacity. | Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
class HGRN2GatedDeltaNet(BaseAttention)
  doc: HGRN-2 Gated DeltaNet Hybrid Architecture. | Mathematical Mechanism:
__init__(config, rank, d_model, num_heads, comp_config, old_attn, lora_rank, lora_alpha)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  self.out_proj.weight.copy_(w_o.reshape(self.out_proj.weight.shape))
  self.out_proj.bias.copy_(b_o.reshape(self.out_proj.bias.shape))
features: causal,delta,distill,gating,lora,peft,rank
loc: 373