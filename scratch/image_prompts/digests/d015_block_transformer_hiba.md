# d015 block_transformer_hiba  [attention]
aliases: block_transformer_hiba
file: Codes/framework/attentions/block_transformer_hiba.py  doc: Docs/content/attentions/block_transformer_hiba.md
doc: Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA). | Production-hardened implementation supporting: | 1. Robust Pretrained Weight Extraction across DistilBERT, RoBERTa/BERT nested encoders, | GPT-2 (fused Conv1D), and Qwen/LLaMA/Mistral.
class LoRALinear(Module)
  doc: Linear projection layer equipped with frozen base weights and trainable residual LoRA adapters. | Base path:
class BlockTransformerHIBA(BaseAttention)
  doc: Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA). | Hardened for production use across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA/Mistral.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, block_size, top_k_inter, rectification_interval, layer_id, layer_idx, old_attn, pretrained_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  attn = attn.masked_fill(pad_mask, float("-inf"))
  probs = F.softmax(attn, dim=-1)
  return torch.matmul(probs.to(v.dtype), v)
  scores = torch.matmul(q_b.float(), k_neighbors.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(pad_mask, float("-inf"))
  scores = scores.masked_fill(sparse_mask == 0, float("-inf"))
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation,sparsity,topk
loc: 724