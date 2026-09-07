# d190 mgie_budget  [compressor]
aliases: mgie_budget
file: Codes/framework/compression_engine/mgie_svd_compressor.py  doc: NONE
doc: MGIE-SVD with Total Rank Budget: Global Entropy-Proportional Rank Allocation. | Registered as 'mgie_budget'.
class MGIESVDBudgetCompressor(BaseCompressor)
  doc: MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation.
__init__(model, total_rank_budget, rank, max_error, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, min_rank, target_modules)
features: entropy,gqa,lora,lowrank,peft,rank,svd
loc: 364