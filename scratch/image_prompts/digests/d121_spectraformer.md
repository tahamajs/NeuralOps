# d121 spectraformer  [attention]
aliases: spectraformer,spectraformer_attention
file: Codes/framework/compression_engine/spectraformer_attention.py  doc: Docs/content/compressions/spectraformer_attention.md
doc: Spectraformer: Random Feature Linear Attention with Flexible Kernel Approximation. | Replaces softmax attention with random orthogonal feature projections, | enabling true linear O(N·r·d) time and memory complexity. | Production-Hardened with:
class LoRALinear(Module)
  doc: Linear layer augmented with optional trainable low-rank adaptation.
class SpectraformerAttention(BaseAttention)
  doc: Spectraformer Attention: Exact linear-time attention via random feature kernelization.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  Q_proj = F.relu(torch.matmul(q, R)) * scale
  K_proj = F.relu(torch.matmul(k, R)) * scale
  state = state + torch.matmul(kt.transpose(-2, -1), vt)  # (B, H, r, d)
  z_state = z_state + kt.transpose(-2, -1)                # (B, H, r, 1)
  num_t = torch.matmul(qt, state)                         # (B, H, 1, d)
shapes: [B, H, 1, 1 [B, H, 1, d [B, H, N, 1 [B, H, N, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Recurrent linear causal attention: exact O(N·r·d) computation
  Bidirectional linear attention: exact O(N·r·d) computation
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 654