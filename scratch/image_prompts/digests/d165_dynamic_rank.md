# d165 dynamic_rank  [compressor]
aliases: dynamic_entropy,dynamic_entropy_svd,dynamic_rank
file: Codes/framework/compression_engine/dynamic_rank.py  doc: Docs/content/compressions/dynamic_entropy.md
doc: Dynamic Entropy Singular Value Decomposition (Dynamic Entropy SVD) Compressor. | Implements singular entropy computation over empirical layer activation covariances: | $$H(\Sigma_l) = -\sum_{i=1}^{d} p_i \ln(p_i), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$ | Dynamically budgets layer ranks $r_l \in [r_{\min}, r_{\max}]$ proportional to singular entropy,
class DynamicSVDLinear(Module)
  doc: True Low-Rank Factorized Linear Layer executing as: | y = (x @ V.T) @ U.T + bias
class DynamicEntropySVDCompressor(BaseCompressor)
  doc: Dynamic Entropy SVD Compressor. | 1. Computes singular entropy H(Sigma_l) on layer activation covariances or weights.
class DynamicEntropySVDCompressorRegistered(DynamicEntropySVDCompressor)
class EntropyRankAllocatorRegistered(DynamicEntropySVDCompressor)
__init__(in_features, out_features, rank, bias, dtype, device)
forward(x)
__init__(model, min_rank, max_rank, target_param_ratio, total_rank_budget, target_rank, max_error, target_attention_only, fallback_to_original, calibration_loader, calibration_texts, target_modules)
tensor_ops:
  return torch.matmul(self.U, self.V)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  x_flat = x.reshape(-1, x.shape[-1]).float()
parameters: torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.empty(out_features, dtype=dtype, device=device
formulas/comments:
  Normalized singular probability mass function p_i = sigma_i / sum(sigma_j)
  RESEARCH FIX: Activation-aware whitening SVD with eigenvalue clamping lambda_i >= 1e-4
  Eigen-decomposition with regularized eigenvalue clamping >= 1e-4
  Freeze uncompressed layers when target_attention_only=True
features: calib,entropy,gating,lora,peft,pool,rank,rope,svd
loc: 877