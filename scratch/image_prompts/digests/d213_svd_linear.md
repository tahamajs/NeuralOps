# d213 svd_linear  [compressor]
aliases: svd_linear,svd_linear_compressor
file: Codes/framework/compression_engine/svd_linear.py  doc: Docs/content/compressions/svd_linear.md
doc: SVD-based low-rank linear layer replacement with optional Fused SVD kernel acceleration. | Registered as 'svd_linear' and 'svd_linear_compressor'.
class SVDLinearLayer(Module)
  doc: Replace a linear layer with factorized U @ V, keeping top-`rank` components. | Optionally dispatches fused in-SRAM kernel to eliminate memory roundtrips.
class SVDLinearCompressor(BaseCompressor)
  doc: Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer.
__init__(linear_layer, rank, use_fused_kernel, use_rms_norm, eps, lora_r, lora_alpha)
forward(x)
__init__(model, rank, max_error, fallback_to_original, protect_critical, target_attention_only, lora_r, lora_alpha, skip_attention_qkv, target_modules, cfg)
tensor_ops:
  W_rec = torch.matmul(U_weighted.float(), V_weighted.float())
  return torch.matmul(self.U, self.V)
  x_flat = x.reshape(-1, self.in_features)
  out = out + (self.scaling * lora_out).reshape(*orig_shape[:-1], self.out_features).to(out.dtype)
parameters: U_weighted, requires_grad=requires_grad | V_weighted, requires_grad=requires_grad | torch.ones(self.rank, dtype=dtype, device=device | base_layer.bias.data.clone( | torch.empty((self.lora_r, self.in_features
features: gqa,lora,peft,pool,rank,rope,svd
loc: 386