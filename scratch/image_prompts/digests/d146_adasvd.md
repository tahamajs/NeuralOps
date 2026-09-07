# d146 adasvd  [compressor]
aliases: adasvd
file: Codes/framework/compression_engine/adasvd_compressor.py  doc: NONE
doc: AdaSVD: Adaptive SVD with per-layer importance allocation and iterative ALS refinement. | Registered as 'adasvd'. Production hardened with reconstruction validation, fallback, | Conv1D support, rank reduction bounds, and comprehensive per-layer stats.
class AdaSVDCompressor(BaseCompressor)
  doc: Adaptive SVD Compressor (AdaSVD). | Assigns SVD ranks dynamically across layers proportional to spectral importance
__init__(model, rank, target_param_ratio, num_iters, max_error, fallback_to_original, target_attention_only, min_rank, target_modules, comp_config)
tensor_ops:
  recon_0 = torch.matmul(A_0, B_0)
  recon = torch.matmul(A, B)
formulas/comments:
  Enforce parameter reduction bound: r * (in_f + out_f) <= 0.75 * in_f * out_f
features: adaptive,gating,lora,lowrank,peft,pool,rank,svd
loc: 488