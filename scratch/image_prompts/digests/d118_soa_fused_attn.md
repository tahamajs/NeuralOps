# d118 soa_fused_attn  [attention]
aliases: soa_fused_attention,soa_fused_attn,soa_fused_module,soafusedattention
file: Codes/framework/attentions/SOAFusedAttention.py  doc: Docs/content/attentions/SOAFusedAttention.md
doc: SOA Fused Attention: Low-Rank Latent Space Multi-Head Attention with Zero-Shot SVD Initialization – Production-Hardened. | Factorizes dense Q, K, and V linear projection operators into separate low-rank bottleneck pairs | initialized from truncated Singular Value Decomposition (SVD) of source attention weights. | Supports:
class LoRALinear(Module)
class FactorizedLinear(Module)
class SOAFusedAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(down, up)
forward(x)
__init__(config, model_config, k_rank, rank, dropout, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  return torch.matmul(self.up.weight, self.down.weight)
  fnorm = torch.linalg.norm(torch.matmul(up.weight, down.weight).float()).item()
  q = self.q_up(self.q_down(hidden_states)).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.k_up(self.k_down(hidden_states)).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.v_up(self.v_down(hidden_states)).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
  probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
shapes: [N, N
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation,svd
loc: 526