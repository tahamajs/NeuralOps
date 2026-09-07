# d014 bam_umoe  [attention]
aliases: bam_umoe
file: Codes/framework/attentions/bam_umoe.py  doc: Docs/content/attentions/bam_umoe.md
doc: BAM-UMoE: Unified Mixture of Experts Attention Block. | This module implements a production-hardened Unified Mixture of Experts (UMoE) attention | mechanism supporting multi-architecture deployments: DistilBERT, RoBERTa/BERT, GPT-2, | and Qwen/LLaMA/Mistral (GQA + partial RoPE).
class LoRALinear(Module)
  doc: Linear projection layer equipped with frozen base weights and trainable residual LoRA adapters. | Base path:
class UnifiedMoE(Module)
  doc: Vectorized Mixture of Experts (MoE) layer with top-k gating, auxiliary load-balancing loss, | and step-0 representation preservation.
class BAMUMoEBlock(BaseAttention)
  doc: BAM Unified Mixture of Experts Attention Block. | Hardened for cross-architecture deployment across DistilBERT, RoBERTa/BERT, GPT-2,
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(dim, num_experts, top_k, output_dim, noise_std, expert_multiplier)
forward(x)
__init__(config, rank, num_experts, top_k, layer_id, num_layers, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  scores = F.softmax(topk_vals, dim=-1)  # (B, L, k)
  router_probs = F.softmax(logits.float(), dim=-1).view(-1, self.num_experts)  # (B*L, E)
  density = expert_mask.sum(dim=2).view(-1, self.num_experts).mean(dim=0)  # (E,)
  topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim),
  w = w.reshape(w.size(0), -1)
  q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_lin(kv_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_lin(kv_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
shapes: [B*L, E [B, L, E [B, L, k [B, L, k, E
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor([-3.0]
features: calib,causal,distill,gating,gqa,lora,moe,peft,rank,rope,rope_partial,rotation,topk
loc: 711