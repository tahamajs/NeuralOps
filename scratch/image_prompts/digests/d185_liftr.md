# d185 liftr  [compressor]
aliases: liftr,liftr_compressor
file: Codes/framework/compression_engine/liftr.py  doc: Docs/content/compressions/liftr.md
doc: LIFTR – Linearized Function Trick Compressor. | Registered in COMPRESSOR_REGISTRY as 'liftr' and 'liftr_compressor'. | Mathematical Formulation: | 1. Activation Function Linearization:
class _LinearizedGELU(Module)
  doc: Linear approximation of GELU / SiLU: f(x) ≈ alpha * x.
class _LinearizedAttentionWrapper(Module)
  doc: Wraps attention modules to replace quadratic softmax attention with sequence pooling. | Preserves all original layer dimensions, handles Conv1D / Linear, and return signature.
class LIFTRCompressor(BaseCompressor)
  doc: LIFTR Compressor: Linearizes activations and attention mechanisms with output-level calibration validation.
__init__(alpha)
forward(x)
__init__(original_attn, layer_name)
forward(hidden_states, query, key, value, mask)
__init__(model, replace_attention, gelu_coefficient, validation_calibration, calibration_threshold, target_attention_only)
tensor_ops:
  context = v.float().mean(dim=1, keepdim=True).expand(v.shape[0], v.shape[1], o_in).to(x.dtype)
  context = v.float().mean(dim=1, keepdim=True).expand(v.shape[0], v.shape[1], o_in).to(x.dtype)
  _, _, v = torch.split(qkv, hidden_size, dim=-1)
  if any(part in name.split(".") for part in ["lora_A", "lora_B"]):
features: calib,delta,distill,lora,pool
loc: 336