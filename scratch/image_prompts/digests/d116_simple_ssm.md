# d116 simple_ssm  [attention]
aliases: simple_ssm
file: Codes/framework/attentions/simple_ssm.py  doc: Docs/content/attentions/simple_ssm.md
doc: Simple State Space Model (SimpleSSM). | Lightweight bidirectional/causal selective state space recurrence block featuring symmetric depthwise 1D convolution, | input-dependent discretization timescales (Delta t), diagonal state transitions (A), multiplicative gating, | and pretrained weight transfer.
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class SimpleSSM(BaseAttention)
  doc: Bidirectional / Causal Simple State Space Model (SimpleSSM) Module. | Features input-dependent discretization timescales (Delta t), diagonal state transitions (A),
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, d_model, d_state, d_conv, expand, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  log_dA = -dt * A.view(1, 1, D)
  C_tr = C.transpose(1, 2)  # [B, D, L]
  u_tr = u.transpose(1, 2).unsqueeze(-1)  # [B, D, L, 1]
  h_fwd = torch.matmul(K_fwd, u_tr).squeeze(-1).transpose(1, 2)  # [B, L, D]
  h_bwd = torch.matmul(K_bwd, u_tr).squeeze(-1).transpose(1, 2)
  out_fwd = (h_fwd * self.D.view(1, 1, D)).to(dtype=x_ssm.dtype)
  out_bwd = (h_bwd * self.D.view(1, 1, D)).to(dtype=x_ssm.dtype) if h_bwd is not None else None
  hidden_states = hidden_states.contiguous()
shapes: [1, 1, D [B, D, L [B, D, L, 1 [B, D, L, L [B, L, D
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank | torch.log(torch.arange(1, d_state + 1, dtype=torch.float32
formulas/comments:
  log_dA = -dt * A: [B, L, D]
  Pairwise log-difference matrix: C_diff[b, d, i, j] = C[b, i, d] - C[b, j, d]
features: causal,chunked,decay,distill,gating,gqa,lora,rank,rope,ssm
loc: 481