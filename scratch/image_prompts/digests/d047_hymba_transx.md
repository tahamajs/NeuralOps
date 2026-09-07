# d047 hymba_transx  [attention]
aliases: hymba_transx
file: Codes/framework/attentions/hymba_transx.py  doc: Docs/content/attentions/hymba_transx.md
doc: HymbaTransX: Hybrid Differential Transformer & State Space Model Block (Production-Hardened). | Differential attention with rotary embeddings coupled with a State Space Model branch. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class HeadRoPE(Module)
class LoRALinear(Module)
class HymbaTransXBlock(BaseAttention)
  doc: HymbaTransX: Hybrid Differential Transformer & State Space Model Block.
__init__(head_dim, max_seq_len, base)
forward(x)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, num_heads, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  freqs = torch.einsum("i,j->ij", t, self.inv_freq)
  w_q = w_q.reshape(w_q.size(0), -1)
  w_k = w_k.reshape(w_k.size(0), -1)
  w_v = w_v.reshape(w_v.size(0), -1)
  w_o = w_o.reshape(w_o.size(0), -1)
  q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k1 = self.W_k1(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor(0.5
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 558