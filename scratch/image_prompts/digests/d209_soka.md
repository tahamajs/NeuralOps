# d209 soka  [compressor]
aliases: soka,soka_compressor
file: Codes/framework/compression_engine/soka_compressor.py  doc: Docs/content/compressions/soka_compressor.md
doc: SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor. | Registered in COMPRESSOR_REGISTRY as 'soka'. | Mathematical Formulation: | 1. Weight Matrix Factorization via SVD:
class LowRankLinear(Module)
  doc: Factorized Linear Layer executing as y = (x @ V) @ U^T + b. | Reduces parameter storage from (d_in * d_out) to r * (d_in + d_out).
class LowRankCompressor(BaseCompressor)
  doc: SoKA Low-Rank Linear Compressor. | Replaces dense nn.Linear and Conv1D layers with factorized LowRankLinear layers.
__init__(in_features, out_features, rank, bias, device, dtype)
forward(x)
__init__(model, rank, energy_threshold, max_error, target_attention_only, fallback_to_original, target_modules)
tensor_ops:
  return torch.matmul(self.U, self.V.t())
  bottleneck = torch.matmul(x, self.V)  # [..., rank]
  out = torch.matmul(bottleneck, self.U.T)  # [..., out_features]
parameters: torch.empty((in_features, self.rank | torch.empty((out_features, self.rank | torch.zeros(out_features, device=device, dtype=dtype
formulas/comments:
  Assign factorized weights: W ≈ (U_r sqrt(S)) @ (sqrt(S) V_r^T)
features: lora,lowrank,peft,pool,rank,rope,svd
loc: 321