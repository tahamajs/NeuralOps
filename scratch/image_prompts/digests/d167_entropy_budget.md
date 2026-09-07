# d167 entropy_budget  [compressor]
aliases: entropy_budget,entropy_budget_allocation
file: Codes/framework/compression_engine/entropy_budget_allocation.py  doc: Docs/content/compressions/entropy_budget_allocation.md
doc: EntropyBudgetAllocation: global rank assignment via whitened activation entropy. | Registered as 'entropy_budget' and 'entropy_budget_allocation'.
class EntropyBudgetCompressor(BaseCompressor)
  doc: Entropy Budget Low-Rank SVD Compressor. | Computes activation covariance and whiten weights, calculates per-component
__init__(model, calibration_loader, total_budget_ratio, rank, eps, max_error, target_attention_only, fallback_to_original, target_modules, cfg)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  x_flat = x.reshape(-1, x.shape[-1]).float()
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  W_recon = torch.matmul(U_r.float(), V_r.float())
features: calib,entropy,gating,lora,lowrank,peft,pool,rank,svd
loc: 467