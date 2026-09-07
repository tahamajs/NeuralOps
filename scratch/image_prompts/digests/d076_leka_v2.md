# d076 leka_v2  [attention]
aliases: leka_v2,leka_v2_attention
file: Codes/framework/attentions/leka_v2.py  doc: Docs/content/attentions/leka_v2.md
doc: LEKAv2: Linear-time Efficient Kernel Attention with Landmark Distillation (Hardened). | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT) | - Trainable LoRA adapters with frozen base weights
class LoRALinear(Module)
class LEKAv2Attention(BaseAttention)
  doc: LEKAv2: Linear-time Efficient Kernel Attention with Landmark Distillation.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, comp_config, layer_idx, total_layers, bert_mode, rank, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  proj1 = torch.matmul(x, W1)
  proj2 = torch.matmul(x, W2)
  x = x.view(new_x_shape)
  return x.permute(0, 2, 1, 3)
  K_land = torch.gather(K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim))
  V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
  w = w.reshape(w.size(0), -1)
  q = self.query(hidden_states).view(B, N, H, self.head_dim).transpose(1, 2)
shapes: [1, 1, N, 1 [1, H, 1, 1 [B, H, 1, L [B, H, L
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.eye(self.head_dim
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation,topk,window
loc: 599