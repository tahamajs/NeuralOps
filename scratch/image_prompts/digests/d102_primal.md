# d102 primal  [attention]
aliases: primal,primal_attention
file: Codes/framework/attentions/primal_attention.py  doc: Docs/content/attentions/primal_attention.md
doc: Primal Attention: Self-Attention through Asymmetric Kernel SVD in Primal Representation – Production-Hardened. | Nyström-inspired low-rank landmark approximation of self-attention using segment pooling landmarks | and iterative Schulz pseudoinverse refinement. | Supports:
class LoRALinear(Module)
class PrimalAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, pinv_iters, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  K = self.key(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  V = self.value(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  Q_land = Q.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(dim=3)
  K_land = K.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(dim=3)
  F1 = torch.matmul(Q.float(), K_land.float().transpose(-2, -1)) * scale
  F2 = torch.matmul(Q_land.float(), K_land.float().transpose(-2, -1)) * scale
  F3 = torch.matmul(Q_land.float(), K.float().transpose(-2, -1)) * scale
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Iterative Schulz refinement: X_{t+1} = X_t (2I - F2 X_t)
features: causal,distill,gqa,lora,peft,pool,rank,rope,rotation,svd
loc: 513