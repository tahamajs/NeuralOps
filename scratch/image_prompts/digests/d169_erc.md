# d169 erc  [compressor]
aliases: erc,erc_svd
file: Codes/framework/compression_engine/erc_svd.py  doc: Docs/content/compressions/erc_svd.md
doc: ERC-SVD: Error-Controlled SVD with Bounded Frobenius Truncation Error. | Registered as 'erc' and 'erc_svd'. Production hardened for multi-architecture | compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support, | parameter bounds, and safe fallback.
class ERCCompressor(BaseCompressor)
  doc: Error-Controlled SVD (ERC-SVD) Compressor. | Finds the minimal rank r for each linear layer such that the relative Frobenius reconstruction
__init__(model, allowed_error, max_error, fallback_to_original, target_attention_only, rank, max_retries, target_modules, cfg)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  W_recon = torch.matmul(U_r.float(), V_r.float())
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
features: distill,gating,lora,lowrank,peft,pool,rank,svd
loc: 389