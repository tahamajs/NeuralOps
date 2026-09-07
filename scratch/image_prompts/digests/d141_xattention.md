# d141 xattention  [attention]
aliases: x_attention,xattention
file: Codes/framework/attentions/xattention.py  doc: Docs/content/attentions/xattention.md
doc: XAttention: Block Sparse Attention with Coarse-to-Fine Routing. | Partitions sequence tokens into non-overlapping blocks, evaluates coarse block-level routing, | and computes unified normalized fine-grained attention across top-k candidate blocks, with multi-architecture | weight transfer, trainable LoRA residual adapters, GQA, RoPE, and causal decoding.
class LoRALinear(Module)
class XAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, block_size, sparsity, lora_rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
  Q_blocks = Q_pad.view(B, H, num_q_blocks, bsz, d).mean(dim=3)
  K_blocks = K_pad.view(B, H, num_k_blocks, bsz, d).mean(dim=3)
  coarse_scores = torch.matmul(Q_blocks, K_blocks.transpose(-1, -2)) * scale
  coarse_scores = coarse_scores.masked_fill(~causal_block_mask, -1e9)
shapes: [B, 1, 1, 1 [B, H, 1, 1 [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Apply RoPE: Priority 1 = position_embeddings, Priority 2 = rotary_emb fallback
features: causal,distill,gqa,lora,rank,rope,rotation,sparsity,topk
loc: 622