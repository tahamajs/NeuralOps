# d025 expert_choice_moe  [attention]
aliases: expert_choice_moe,expert_choice_moe_attention
file: Codes/framework/attentions/expert_choice_moe.py  doc: Docs/content/attentions/expert_choice_moe.md
doc: Expert Choice Mixture of Experts (MoE) Attention Block. | In Expert Choice routing (Zhou et al., 2022), experts select their top-k tokens | rather than tokens selecting top-k experts. This provides guaranteed perfect load balancing | across all experts, dynamic per-token expert allocation, and superior zero-shot transfer.
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class ExpertChoiceMoE(Module)
  doc: Expert Choice Routing MoE layer. | Each expert selects top-k tokens with the highest affinity scores.
class ExpertChoiceMoEAttention(BaseAttention)
  doc: Expert Choice MoE Multi-Head Attention Module.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(d_model, n_experts, capacity_factor, noise_std)
forward(x)
__init__(config, rank, n_experts, capacity_factor, pretrained_attn, old_attn, comp_config, layer_id, layer_idx, moe_init_gate)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  x_flat = x.view(-1, D)
  gate_scores = F.softmax(logits.float(), dim=0).to(dtype=logits.dtype)
  prob_mean = F.softmax(logits.float() / 1.0, dim=-1).mean(dim=0)
  return out_flat.view(B, L, D)
  w = w.reshape(w.size(0), -1)
  q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
shapes: [B, L, D
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor(float(moe_init_gate
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,distill,gating,gqa,lora,moe,peft,rank,rope,rope_partial,rotation,topk
loc: 662