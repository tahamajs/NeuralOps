# d072 jamba  [attention]
aliases: jamba,jamba2
file: Codes/framework/attentions/jamba.py  doc: Docs/content/attentions/jamba.md
doc: Jamba-2 Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block (Production-Hardened). | Combines: | - SVD-Mamba selective state space branch | - Multi-Head Attention branch with GQA, RoPE, and causal masking
class LoRALinear(Module)
class SVDMambaSub(Module)
class Jamba2Block(BaseAttention)
  doc: Jamba-2 Hybrid Architecture combining SVD-Mamba, Multi-Head Attention, and Expert Choice MoE.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(d_model, config, rank, is_decoder)
forward(hidden_states, attention_mask)
__init__(config, d_model, n_heads, n_experts, top_k, rank, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  q = self.q_lin(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_lin(x1).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_lin(x1).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
  scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,moe,peft,rank,rope,rotation,ssm,svd,topk
loc: 650