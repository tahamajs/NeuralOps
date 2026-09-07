# d032 fusion_attn  [attention]
aliases: fusion_attn
file: Codes/framework/attentions/fusion_attn.py  doc: Docs/content/attentions/fusion_attn.md
doc: FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module. | Implements hybrid sparse attention with block gating, full-attention short sequence fast-path, | and subspace estimation. | Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class FUSIONATTN(BaseAttention)
  doc: FUSIONATTN: Hybrid Sparse Attention with Block Gating and Subspace Estimation.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, hidden_size, num_heads, block_size, top_k_blocks, rank, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  q_blocks = q_pad.reshape(B, H, num_blocks, self.block_size, D)
  k_blocks = k_pad.reshape(B, H, num_blocks, self.block_size, D)
  v_blocks = v_pad.reshape(B, H, num_blocks, self.block_size, D)
  idx_exp = idx_b.unsqueeze(-1).unsqueeze(-1).expand(B, H, top_k, self.block_size, D)
  k_sel = torch.gather(k_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)
shapes: [N, N
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.ones(1
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,distill,gating,gqa,lora,peft,rank,rope,rope_partial,rotation,sparsity,topk
loc: 634