# d155 coopq  [compressor]
aliases: coopq,coopq_quantizer
file: Codes/framework/compression_engine/coopq_quantizer.py  doc: Docs/content/compressions/coopq_quantizer.md
doc: CoopQ: Cooperative Game-Theoretic Bit Allocation Quantizer with STE and Per-Channel Asymmetric Calibration. | Registered in COMPRESSOR_REGISTRY as 'coopq'. | Mathematical Formulation: | 1. Min Bit-Width Floor Enforcement:
class CoopQSTEFakeQuantize(Function)
  doc: Straight-Through Estimator (STE) fake-quantization autograd function for CoopQ. | Forwards per-channel quantized-dequantized weights and passes gradients through on backward pass.
class CoopQLinear(Module)
  doc: CoopQ Linear layer with per-channel asymmetric quantization and STE backpropagation. | Supports both standard nn.Linear and Conv1D weight representations.
class CoopQQuantizer(BaseCompressor)
  doc: CoopQ: Cooperative Game-Theoretic Bit Allocation Quantization. | Allocates discrete bit-widths (>= 4 bits) across transformer layers
forward(x, scale, zero_point, qmin, qmax)
__init__(in_features, out_features, bias, bits, symmetric, per_channel)
forward(x)
__init__(model, bits, min_bits, max_bits, symmetric, per_channel, use_ste, recalibrate_ln, target_attention_only, tokenizer, calibration_loader, calibration_texts)
tensor_ops:
  w_data = raw_w.t().contiguous()
  w_data = raw_w.contiguous()
  target_mod.weight.data.copy_(W_quant.t().contiguous().to(dtype=raw_w.dtype, device=raw_w.device))
  target_mod.weight.data.copy_(W_quant.contiguous().to(dtype=raw_w.dtype, device=raw_w.device))
parameters: torch.empty(out_features, in_features | torch.empty(out_features
formulas/comments:
  Allowed bit-levels with floor >= self.min_bits (>= 4)
features: calib,gating,lora,peft,pool,quant
loc: 575