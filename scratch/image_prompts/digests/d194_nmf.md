# d194 nmf  [compressor]
aliases: nmf,nmf_compressor
file: Codes/framework/compression_engine/nmf_compressor.py  doc: Docs/content/compressions/nmf_compressor.md
doc: NMF: Non-negative Matrix Factorization Linear Compressor. | Registered in COMPRESSOR_REGISTRY as 'nmf'.
class NMFLinear(Module)
  doc: Linear layer decomposed via Non-negative Matrix Factorization (NMF): | W = H @ V + shift
class NMFCompressor(BaseCompressor)
  doc: Non-negative Matrix Factorization (NMF) Compressor. | Decomposes linear layer weights into low-rank non-negative factors.
__init__(in_features, out_features, rank, shift, bias, dtype, device, lora_r, lora_alpha)
forward(x)
__init__(model, rank, max_iter, lora_r, lora_alpha, cfg, target_layers, target_attention_only, max_error, fallback_to_original)
tensor_ops:
  return torch.matmul(self.H, self.V) + self.shift
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  return torch.matmul(self.H, self.V) + self.shift
  W = raw_w.t().contiguous()
  HTW = torch.matmul(HT, W_pos)
  HTHV = torch.matmul(torch.matmul(HT, H_init), V_init).clamp_min(1e-10)
  WVT = torch.matmul(W_pos, VT)
parameters: torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.zeros(out_features, dtype=dtype, device=device | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: lora,peft,pool,rank,rope,svd
loc: 424