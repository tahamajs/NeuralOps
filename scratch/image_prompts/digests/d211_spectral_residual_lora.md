# d211 spectral_residual_lora  [compressor]
aliases: spectral_residual_lora
file: Codes/framework/compression_engine/res_svd.py  doc: NONE
doc: Spectral Residual Low-Rank Adaptation (Res-SVD / Spectral Residual LoRA). | Decomposes dense weight matrices into: | 1. Frozen Base Low-Rank Factors: $W_{\text{base}} = U_r \Sigma_r V_r^\top$ (rank $r$, frozen). | 2. Trainable Residual Adapters: Initialized directly from the top-$k$ singular directions of the
class ResSVDLinear(Module)
  doc: Spectral Residual LoRA (Res-SVD) Linear Layer. | Executes forward pass as:
class ResSVDCompressor(BaseCompressor)
  doc: Spectral Residual LoRA (Res-SVD) Compressor. | Replaces standard Linear layers with ResSVDLinear modules, truncating base weights
class SpectralResidualLoRA(ResSVDCompressor)
  doc: PEFT-focused wrapper for Spectral Residual LoRA.
class ResSVDCompressorRegistered(ResSVDCompressor)
class SpectralResidualLoRARegistered(SpectralResidualLoRA)
__init__(in_features, out_features, base_rank, residual_rank, scaling, factorize_base, bias, dtype, device)
forward(x)
__init__(model, base_rank, residual_rank, scaling, factorize_base, freeze_base, target_modules, max_error, fallback_to_original, target_attention_only)
__init__()
tensor_ops:
  base = torch.matmul(self.base_U, self.base_V)
  res = self.scaling * torch.matmul(self.lora_B, self.lora_A)
parameters: torch.empty(self.base_rank, in_features, dtype=dtype, device | torch.empty(out_features, self.base_rank, dtype=dtype, devic | torch.empty(out_features, in_features, dtype=dtype, device=d | torch.empty(self.residual_rank, in_features, dtype=dtype, de | torch.empty(out_features, self.residual_rank, dtype=dtype, d
formulas/comments:
  Scale S_tail by adapter scaling factor gamma: (B @ A) * gamma = E_tail
features: lora,peft,pool,rank,rope,svd
loc: 582