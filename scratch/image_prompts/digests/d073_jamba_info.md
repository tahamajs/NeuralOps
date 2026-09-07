# d073 jamba_info  [attention]
aliases: jamba_info
file: Codes/framework/attentions/jamba_info.py  doc: Docs/content/attentions/jamba_info.md
doc: Information-Bottlenecked Jamba Architecture with Dual Operational Modes (Production-Hardened). | Differential attention with concept-bottlenecked SSM branch. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
class InfoMambaConceptBottleneck(Module)
class JambaInfoBlock(BaseAttention)
  doc: Information-Bottlenecked Jamba Architecture with Dual Operational Modes.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(dim, bottleneck_dim)
forward(x)
__init__(config, rank, mode, sparsity, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w_q = w_q.reshape(w_q.size(0), -1)
  w_k = w_k.reshape(w_k.size(0), -1)
  w_v = w_v.reshape(w_v.size(0), -1)
  w_o = w_o.reshape(w_o.size(0), -1)
  q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k1 = self.W_k1(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k2 = self.W_k2(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation,sparsity,ssm
loc: 541