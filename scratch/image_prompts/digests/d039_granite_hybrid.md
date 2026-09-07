# d039 granite_hybrid  [attention]
aliases: granite_hybrid,granite_hybrid_block
file: Codes/framework/attentions/granite_hybrid.py  doc: Docs/content/attentions/granite_hybrid.md
doc: Granite Hybrid Attention-SSM Layer with MoE Feed-Forward. | Combines Mamba-2 SSM recurrent layers with dense Multihead Attention (MHA) and | Top-k Mixture-of-Experts (MoE) feed-forward, with LoRA adapters and pretrained weight transfer. | Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
class Mamba2(Module)
  doc: Simplified Mamba-2 State Space Model (SSM) block.
class MixtureOfExperts(Module)
  doc: Top-$k$ Token-Choice Mixture-of-Experts (MoE) Layer.
class GraniteHybridBlock(BaseAttention)
  doc: Granite Hybrid Attention-SSM Layer with MoE Feed-Forward and LoRA Adaptation.
__init__(d_model, d_state, d_conv, expand)
forward(x)
__init__(dim, ff_dim, num_experts, top_k)
forward(x)
__init__(config, rank, dim, num_heads, num_experts, top_k, ratio, pretrained_attn, old_attn, lora_rank, lora_alpha)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  x_ssm, z = x_proj.chunk(2, dim=-1)
  x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
  dA_exp_arg = torch.clamp(torch.einsum("bli,in->blin", dt, A), min=-30.0, max=10.0)
  dBX = torch.einsum("bli,blin->blin", x_ssm, dt.unsqueeze(-1))
  out = (h * self.D.view(1, 1, inner_dim, 1)).sum(dim=-1) * F.silu(z)
  weights = F.softmax(weights, dim=-1)
  q = self.q_proj(q_in).view(B, L_q, H, Dh).transpose(1, 2)
  k = self.k_proj(k_in).view(B, L_k, H, Dh).transpose(1, 2)
shapes: [B, L [B, L, E [w, b
parameters: torch.log(torch.rand(inner_dim, d_state | torch.ones(inner_dim
features: causal,chunked,decay,distill,gating,lora,moe,peft,rank,ssm,topk
loc: 497