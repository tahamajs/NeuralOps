# d175 grecodec  [compressor]
aliases: grecodec,grecodec_compressor
file: Codes/framework/compression_engine/grecodec_compressor.py  doc: Docs/content/compressions/grecodec_compressor.md
doc: GreCoDeC: Greedy Coordinate Descent Low-Rank Compressor. | Registered in COMPRESSOR_REGISTRY as 'grecodec'.
class GreCoDeCLinear(Module)
  doc: Greedy Coordinate Descent Factorized Linear Layer with optional LoRA adapter. | W ≈ U @ V
class GreCoDeCCompressor(BaseCompressor)
  doc: GreCoDeC Compressor. | Decomposes linear layers using coordinate descent low-rank factorization.
__init__(in_features, out_features, rank, bias, lora_r, lora_alpha, dtype, device)
forward(x)
__init__(model, rank, max_iter, lora_r, lora_alpha, max_error, target_attention_only, fallback_to_original, skip_attention_qkv, cfg, target_layers)
tensor_ops:
  return torch.matmul(self.U, self.V)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  return torch.matmul(self.U, self.V)
  VVt = torch.matmul(V_mat, V_mat.t())
  WV_t = torch.matmul(W_float, V_mat.t())
  UtU = torch.matmul(U_mat.t(), U_mat)
  UtW = torch.matmul(U_mat.t(), W_float)
parameters: torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.zeros(out_features, dtype=dtype, device=device | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
formulas/comments:
  Update U given V: min_U ||W - U V||_F^2 => U = W V^T (V V^T)^{-1}
  Update V given U: min_V ||W - U V||_F^2 => V = (U^T U)^{-1} U^T W
features: gating,gqa,lora,peft,pool,rank,rope,svd
loc: 481