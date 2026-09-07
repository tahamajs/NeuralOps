# d201 rotational_slicegpt  [compressor]
aliases: rotational_slicegpt,rotational_slicegpt_int4
file: Codes/framework/compression_engine/rotational_slicegpt_int4.py  doc: Docs/content/compressions/rotational_slicegpt_int4.md
doc: Rotational SliceGPT INT4 Compressor: Rotational Invariance, Subspace Slicing, and Second-Order INT4 Quantization. | Registered in COMPRESSOR_REGISTRY as 'rotational_slicegpt_int4' and 'rotational_slicegpt'. | Mathematical Formulation: | 1. Orthogonal Coordinate Rotation:
class AstroQuantizer()
  doc: Astro Non-Uniform Quantization Codebook mapping supporting 2, 4, and 8 bits. | Maps floating-point parameters to non-linear centroids optimized for Gaussian weight distributions.
class RotationalSliceGPTINT4Linear(Module)
  doc: A linear layer that applies the compressed Rotational-SliceGPT-INT4 weights. | Maintains equivalent outputs to W @ X by evaluating: W_quant @ (Q P_r)^T X + bias.
class RotationalSliceGPTINT4Compressor(BaseCompressor)
  doc: Rotational-SliceGPT-INT4 Hybrid Compression Engine. | Combines orthogonal rotation, variance-based coordinate slicing, and
__init__(bits)
__init__(W_quant, Q, P_r, bias)
forward(x)
__init__(model, r_ratio, lambda_reg, slice_ratio, quant_bits, max_error, fallback_to_original, tokenizer, calibration_loader, calibration_texts, device, target_modules)
tensor_ops:
  Q_P_r = torch.matmul(Q, P_r)  # [D_in, r]
  return torch.matmul(self.W_quant, self.Q_P_r.T)
  x_proj = torch.matmul(x, self.Q_P_r)
  y = torch.matmul(x_proj, self.W_quant.T)
  W_tilde = torch.matmul(W_fp32, Q)
  X_tilde = torch.matmul(Q.T, X_fp32)
  W_sliced = torch.matmul(W_tilde, P_r)
  X_sliced = torch.matmul(P_r.T, X_tilde)
parameters: bias.clone(
formulas/comments:
  1. Orthogonal Rotation: \tilde{W} = W Q, \tilde{X} = Q^T X
features: calib,gating,lora,peft,pool,quant,rope,rotation
loc: 529