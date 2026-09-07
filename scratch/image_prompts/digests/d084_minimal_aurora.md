# d084 minimal_aurora  [attention]
aliases: minimal_aurora
file: Codes/framework/attentions/minimal_aurora.py  doc: Docs/content/attentions/minimal_aurora.md
class FactorisedLinear(Module)
  doc: SVD-Factorized Linear Transformation with LoRA Residual Branch.
class MinimalAURORAAttention(BaseAttention)
  doc: MinimalAURORA: Energy-Preserving SVD-Factorized Attention. | Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
__init__(in_features, out_features, rank, bias, energy, lora_rank, lora_alpha)
forward(x)
__init__(config, model_config, energy, dropout, pretrained_attn, rank, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  W = W.view(self.out_features, rep, -1).mean(dim=1)
  bias = bias.view(self.out_features, rep).mean(dim=1)
  base_w = torch.matmul(self.up.weight, self.down.weight)
  Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, N_kv, self.num_kv_heads, D).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, N_kv, self.num_kv_heads, D).transpose(1, 2)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  attn_weights = F.softmax(scores, dim=-1).to(dtype)
shapes: [B, N, H * D [B, N, H, D [w, b
parameters: torch.randn(lora_rank, in_features | torch.zeros(out_features, lora_rank | torch.tensor(2.0
features: causal,distill,gating,gqa,lora,rank,rope,rotation,svd
loc: 519