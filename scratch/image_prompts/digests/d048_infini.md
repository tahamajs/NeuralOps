# d048 infini  [attention]
aliases: infini,infini_attention
file: Codes/framework/attentions/infini_attention.py  doc: Docs/content/attentions/infini_attention.md
doc: Infini-Attention with Compressive Memory and Sliding Window (Production-Hardened). | Combines local sliding-window attention with learned compressive memory. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
class InfiniAttention(BaseAttention)
  doc: Infini-Attention with Compressive Memory and Sliding Window.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, compress_dim, local_window, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2)
  mem = self.memory.expand(B, -1, -1, -1)
  mem_scores = torch.matmul(q.float(), mem.float().transpose(-2, -1)) / math.sqrt(Dh)
  mem_attn = torch.softmax(mem_scores, dim=-1)
  mem_out = torch.matmul(mem_attn, mem.float()).to(v.dtype)
shapes: [B, L, H, Dh
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.randn(1, self.num_heads, compress_dim, self.head_dim
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation,window
loc: 512