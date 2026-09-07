# d096 nystrom  [attention]
aliases: nystrom,nystrom_attention
file: Codes/framework/attentions/nystrom_attention.py  doc: Docs/content/attentions/nystrom_attention.md
doc: Nyströmformer Attention – Production-Hardened. | Nyström-based low-rank kernel approximation with iterative pseudoinverse. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
class NystromAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, num_landmarks, pinv_iterations, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  V = alpha * A.transpose(-2, -1)
  V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
  Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
  K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)
  K_flat = K.reshape(B * H, N, d).transpose(1, 2)
shapes: [B * H, N, d [B, N, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Associative linear multiplication: O(N * m * d)
features: adaptive,causal,distill,gqa,lora,peft,pool,rank,rope,rotation
loc: 532