# d204 salr  [compressor]
aliases: salr
file: Codes/framework/compression_engine/salr.py  doc: Docs/content/compressions/salr.md
doc: SALR: Structure-Aware / Sparse Low-Rank compression combining SVD with bottleneck pruning. | Registered in COMPRESSOR_REGISTRY as 'salr', 'salr_factory', and 'salr_compressor_factory'.
class SALRLayer(Module)
  doc: Structure-Aware Low-Rank + Sparse Layer (SALR). | Combines truncated SVD rank reduction with sparse singular component selection
class SALRCompressor(BaseCompressor)
  doc: SALR Compressor: Structure-Aware Low-Rank and Sparse Pruning. | Factorizes dense nn.Linear or Conv1D layers into symmetric low-rank factors
__init__(in_features, out_features, rank, sparsity, bias, dtype, device, lora_r, lora_alpha)
forward(x)
__init__(model, rank, sparsity, min_rank, skip_attention_qkv, target_attention_only, use_lora, lora_r, lora_alpha, target_modules, max_error, fallback_to_original)
tensor_ops:
  return torch.matmul(self.U, V_masked)
  W = raw_w.t().contiguous().float()
  W = raw_w.contiguous().float()
parameters: torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.empty(out_features, dtype=dtype, device=device | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: gqa,lora,lowrank,peft,pool,rank,rope,sparsity,svd
loc: 479