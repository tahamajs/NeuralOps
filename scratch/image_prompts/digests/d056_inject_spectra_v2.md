# d056 inject_spectra_v2  [attention]
aliases: inject_spectra_v2
file: Codes/framework/attentions/spectra_v2.py  doc: NONE
doc: SPECTRA-v2: Sparse Per-Head Efficient Context Transformer with Rectified Attention – Production-Hardened. | Hybrid sparse-attention architecture unifying intra-block local attention, relative-position-biased | inter-block top-k routing, per-head adaptive attention spans, and periodic global Random Fourier Feature (RFF) rectification. | Supports:
class SpectraConfig()
class LoRALinear(Module)
class SpectraV2Attention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, model_config, spectra_cfg, layer_idx, total_layers, bert_mode, pretrained_attn, rank)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  K = self.key(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  V = self.value(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
  attn_weights = attn_weights * head_mask.view(1, self.num_heads, 1, 1).to(attn_weights.dtype)
  attn_weights = attn_weights * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(attn_weights.dtype)
  context = torch.matmul(attn_weights, V).permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
shapes: [N, N [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: adaptive,causal,distill,gqa,lora,peft,rank,rope,rotation,sparsity,topk
loc: 656