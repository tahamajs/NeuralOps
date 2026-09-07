# d122 spectral_mamba  [attention]
aliases: spectral_mamba
file: Codes/framework/attentions/spectral_mamba.py  doc: Docs/content/attentions/spectral_mamba.md
doc: Spectral-Mamba Attention Module: Hybrid Attention-State Space Module with Spectral Channel Modulation – Production-Hardened. | Combines pretrained multi-head attention with a Spectral Mamba SSM branch via learned adaptive gating. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
class SpectralMambaConfig()
class SpectralMamba(Module)
class SpectralMambaAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(d_model, d_state)
forward(x)
__init__(config, rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class Mamba(Module)
__init__(d_model)
forward(x)
tensor_ops:
  x1, x2 = self.in_proj(x).chunk(2, dim=-1)
  q = self.q_lin(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.k_lin(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.v_lin(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
  attn_weights = attn_weights * head_mask.view(1, self.num_heads, 1, 1).to(attn_weights.dtype)
  attn_weights = attn_weights * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(attn_weights.dtype)
shapes: [B, N, 1 [N, N [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.zeros(d_model
features: adaptive,causal,chunked,distill,gating,gqa,lora,peft,rank,rope,rotation,ssm
loc: 681