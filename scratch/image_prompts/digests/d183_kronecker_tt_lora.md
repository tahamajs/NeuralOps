# d183 kronecker_tt_lora  [compressor]
aliases: kronecker_tt_lora
file: Codes/framework/compression_engine/kronecker_tt_lora.py  doc: Docs/content/compressions/kronecker_tt_lora.md
doc: Kronecker-TT-LoRA Hybrid Compression Engine Adapter. | Implements a hybrid parameter-efficient fine-tuning (PEFT) and compression mechanism | by decomposing weight updates into a Kronecker product multiplied element-wise by a | Tensor-Train (TT) core tensor with dynamic singular value entropy adaptation,
class KroneckerTTLoRALayer(Module)
  doc: Kronecker-TT-LoRA Hybrid Compression Layer with Symmetric LoRA Bypass and Shape Padding. | Combines Kronecker product structure (A1 ⊗ A2) with a 3-core Tensor-Train tensor T(G1, G2, G3)
class KroneckerTTLoRACompressor(BaseCompressor)
  doc: Compressor wrapper to apply Kronecker-TT-LoRA adapters to linear and Conv1D layers. | Inherits from BaseCompressor.
__init__(base_layer, m1, n1, m2, n2, tt_shapes, tt_ranks, gamma, lora_r, lora_alpha, r_min, r_max)
forward(x)
__init__(model, rank, gamma, lora_r, lora_alpha, max_error, target_attention_only, fallback_to_original, skip_attention_qkv, cfg)
tensor_ops:
  flat = flat.T.contiguous()
  return flat.view(shape).to(device=device, dtype=dtype)
  g1 = self.G1.view(self.tt_shapes[0], self.tt_ranks[1])
  g2 = self.G2.view(self.tt_ranks[1], self.tt_shapes[1] * self.tt_ranks[2])
  core12 = torch.matmul(g1, g2).view(self.tt_shapes[0] * self.tt_shapes[1], self.tt_ranks[2])
  g3 = self.G3.view(self.tt_ranks[2], self.tt_shapes[2])
  tt_tensor = torch.matmul(core12, g3)
  return tt_tensor.view(self.padded_out_features, self.padded_in_features)
shapes: [d1 * d2 * d3, [d1 * d2, d3
parameters: _init_orthogonal_spectral((m1, n1 | _init_orthogonal_spectral((m2, n2 | _init_orthogonal_spectral(                 (1, tt_shapes[0], | _init_orthogonal_spectral(                 (tt_ranks[2], tt_ | torch.zeros((self.lora_r, self.in_features
features: entropy,gating,gqa,kronecker,lora,peft,pool,rank,rope,svd,tt
loc: 643