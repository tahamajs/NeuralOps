# d024 eat  [attention]
aliases: eat,eat_attention
file: Codes/framework/attentions/eat.py  doc: Docs/content/attentions/eat.md
doc: EAT: Efficient Attention Transformer with Progressive Token Pruning. | Implements production-hardened attention-level efficient multi-head self-attention | with optional progressive token pruning and zero-shot weight inheritance across | DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures.
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class EATBlock(BaseAttention)
  doc: Efficient Attention Transformer (EAT) Attention Module with Progressive Token Pruning. | Hardened for multi-arch deployment (DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA).
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, n_heads, prune_layer, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_lin(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_lin(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(pad_mask, float("-inf"))
  probs = F.softmax(scores, dim=-1)
  head_mask = head_mask.view(1, -1, 1, 1)
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 553