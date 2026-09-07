# d163 direct_svd  [compressor]
aliases: direct_svd,direct_svd_compressor
file: Codes/framework/compression_engine/direct_svd.py  doc: Docs/content/compressions/direct_svd.md
doc: Direct SVD Compressor: Truncated SVD low-rank factorized layer replacement. | Registered as 'direct_svd'. Production hardened for multi-architecture compatibility | (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support, rank bounds, and safe fallback.
class DirectSVDCompressor(BaseCompressor)
  doc: Direct SVD Compressor. | Performs direct truncated SVD on target linear and Conv1D layers:
__init__(model, rank, min_rank, rank_ratio, max_error, fallback_to_original, protect_critical, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, max_retries)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  W_recon = torch.matmul(U_r.float(), V_r.float())
formulas/comments:
  Handle freezing of non-target layers when target_attention_only=True
features: distill,gqa,lora,lowrank,peft,pool,rank,svd
loc: 462