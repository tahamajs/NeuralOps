# d055 inject_samba_moe  [attention]
aliases: inject_samba_moe
file: Codes/framework/attentions/samba_moe.py  doc: NONE
doc: Samba-MoE: Hybrid Selective State Space (SAMBA-M3) + REAP-MoE Block. | Combines local sliding-window attention, spectral Mamba SSM, and MoE routing. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA)
class MinimalMoEFallback(Module)
  doc: Fallback MoE layer when REAPMoE cannot be instantiated.
class SambaMoEBlock(BaseAttention)
  doc: SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying.
__init__(d_model, n_experts)
forward(x)
__init__(config, d_model, n_heads, n_experts, window, rank, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  weights = F.softmax(logits, dim=-1)
features: causal,distill,gating,gqa,lora,moe,rank,rope,ssm,window
loc: 242