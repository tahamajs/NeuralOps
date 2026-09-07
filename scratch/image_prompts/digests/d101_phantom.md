# d101 phantom  [attention]
aliases: phantom,phantom_module
file: Codes/framework/attentions/phantom.py  doc: Docs/content/attentions/phantom.md
doc: Phantom: Parameter-Hashed AttentioN with Token-wise Orthogonal Mapping – Production-Hardened. | Combines CUR-matrix factorized linear projections based on empirical statistical leverage scores | with multi-round angular Locality-Sensitive Hashing (LSH) sparse attention and rectified Softpick activation. | Supports:
class PHANTOMConfig()
class FactorisedLinear_CUR(Module)
  doc: Linear projection compressed via leverage-score CUR matrix decomposition with LoRA support.
class PHANTOMAttention(BaseAttention)
__init__(in_features, out_features, cur_rank, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, phantom_cfg, pretrained_attn, bert_mode)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  fallback = F.softmax(x / max(temperature, 1e-4), dim=dim)
  return torch.matmul(self.up.weight, self.down.weight)
  scores_base = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  scores_base = scores_base.masked_fill(~val_mask, -1e4)
  scores_base = scores_base.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
  hq = (torch.matmul(Q_norm, hash_proj) > 0).float() @ bit_shifts  # [B, H, N]
  hk = (torch.matmul(K_norm, hash_proj) > 0).float() @ bit_shifts  # [B, H, N]
  scores_round = scores_base.masked_fill(~match, -1e4)
shapes: [B, 1, 1, N [B, H, N [B, H, N, N [B, N, H, D [N, N [w, b
parameters: torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,hash,lora,peft,rank,rope,rotation,sparsity,svd,topk
loc: 629