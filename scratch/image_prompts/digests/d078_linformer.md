# d078 linformer  [attention]
aliases: linformer,linformer_attention
file: Codes/framework/attentions/linformer_attention.py  doc: Docs/content/attentions/linformer_attention.md
doc: Linformer Attention – Production-hardened with full multi-architecture support. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT) | - Trainable LoRA adapters with frozen base weights
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class LinformerAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, k, max_seq_len, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
  k = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  v = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
  K_proj = torch.matmul(E_act, k)
  V_proj = torch.matmul(F_act, v)
  scores = torch.matmul(q.float(), K_proj.transpose(-2, -1).float()) * scale  # [B, H, N, k]
shapes: [B, H, N, d [B, H, N, k [B, H, k, d [B, N, H, d [k, N
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.empty(self.k_proj_dim, self.max_seq_len
formulas/comments:
  LoRA config (three-branch to handle config=None properly)
  Preservation gate: sigmoid(-3) ≈ 0.047 → output starts as ~95% passthrough of hidden_states.
  sigmoid(-3) ≈ 0.047 at init → 95% passthrough preserves pretrained activations.
features: causal,distill,gating,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 626