# d052 inject_rpc  [attention]
aliases: inject_rpc
file: Codes/framework/attentions/rpc_attention.py  doc: NONE
doc: Random Projection Compressed (RPC) Attention – Production-Hardened. | Employs randomized linear subspace sketching via fixed head-wise Gaussian random projection buffers | to compress query-key similarity evaluations into lower-dimensional manifolds. | Supports:
class LoRALinear(Module)
class RPCAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.W_q(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.W_k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.W_v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  q_sketched = torch.einsum("b h l d, h d r -> b h l r", q, proj)
  k_sketched = torch.einsum("b h l d, h d r -> b h l r", k, proj)
  scores = torch.matmul(q_sketched.float(), k_sketched.float().transpose(-2, -1)) * scale
  probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
  probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 547