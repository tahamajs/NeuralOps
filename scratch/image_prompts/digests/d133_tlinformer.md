# d133 tlinformer  [attention]
aliases: tlinformer,tlinformer_attention
file: Codes/framework/compression_engine/tlinformer_attention.py  doc: Docs/content/compressions/tlinformer_attention.md
doc: TLinFormer: Exact linear‑time attention via low‑rank sequence projections. | Reference: LinFormer (Wang et al., 2020) - adapted for DistilBERT, RoBERTa, GPT-2, and Qwen3.5. | Production-Hardened with: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
  doc: Linear layer augmented with optional trainable low-rank adaptation.
class TLinFormerAttention(BaseAttention)
  doc: Production-Hardened Truncated LinFormer Attention. | Projects key and value sequence length dimensions down to a compressed rank r << N
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  K_proj = torch.matmul(P_k, k_masked)
  V_proj = torch.matmul(P_v, v_masked)
  scores = torch.matmul(q, K_proj.transpose(-2, -1)) * scale
  scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
  hm = hm.view(1, -1, 1, 1)
shapes: [B, 1, N, 1 [B, H, N, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.empty(self.rank, self.max_seq_len
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 642