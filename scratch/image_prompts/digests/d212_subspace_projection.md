# d212 subspace_projection  [compressor]
aliases: subspace_projection,subspace_projection_compressor
file: Codes/framework/compression_engine/subspace_projection.py  doc: Docs/content/compressions/subspace_projection.md
doc: Subspace Projection Compressor: Low-Rank Subspace Projection and Factorization. | Registered in COMPRESSOR_REGISTRY as 'subspace_projection'.
class SubspaceLinear(Module)
  doc: Subspace Projection Factorized Linear Layer with optional LoRA adapter support. | W ≈ B @ A where:
class SubspaceProjectionCompressor(BaseCompressor)
  doc: Subspace Projection Compression Engine. | Compresses neural networks by projecting weight matrices onto their principal
__init__(in_features, out_features, rank, bias, lora_r, lora_alpha, dtype, device)
forward(x)
__init__(model, rank, method, cfg, target_layers, target_attention_only, use_lora, lora_r, lora_alpha, max_error, fallback_to_original)
tensor_ops:
  return torch.matmul(self.B, self.A)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  return torch.matmul(self.B, self.A)
  W = raw_w.t().contiguous()
parameters: torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.zeros(out_features, dtype=dtype, device=device | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: lora,peft,pool,rank,rope,svd
loc: 351