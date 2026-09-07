# d082 mamba  [attention]
aliases: mamba
file: Codes/framework/attentions/mamba.py  doc: Docs/content/attentions/mamba.md
doc: Mamba: Gated State Space Model (Mamba) linear attention replacement (Production-Hardened). | Implements a gated selective state-space projection that is a drop-in | replacement for a standard Multi-Head Attention layer, operating in | strictly O(L) time and space with respect to sequence length, with LoRA adapters,
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class Mamba(BaseAttention)
  doc: Gated State Space Model (Mamba) Linear Layer Module.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, comp_config, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states.view(B, L, H * Dh)
  hidden_states = hidden_states.contiguous()
  pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
  p1, p2 = proj.chunk(2, dim=-1)
  log_weights = log_weights.masked_fill(~causal_mask, float("-inf"))
  weights = torch.softmax(log_weights, dim=1).to(p1.dtype)  # [L_q, L_k, D]
  p1_causal = torch.einsum("ijd,bjd->bid", weights, p1)
shapes: [B, L, 2D [B, L, D [B, L, H * Dh [L, L, D
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.log(torch.arange(1, self.d_model + 1, dtype=torch.floa
formulas/comments:
  Selective SSM decay parameter for causal sequences: A = -exp(A_log) <= 0
  Learned continuous decay: A = -exp(A_log), discrete decay factor gamma = sigmoid(A) in (0, 1)
  Build decay matrix: distance = i - j for i >= j
features: causal,chunked,decay,distill,gating,lora,peft,rank,rope,rotation,ssm
loc: 582