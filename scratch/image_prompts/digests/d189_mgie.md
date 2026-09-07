# d189 mgie  [compressor]
aliases: mgie,mgie_svd
file: Codes/framework/compression_engine/mgie_svd.py  doc: NONE
doc: MGIE-SVD: Maximum Information Entropy SVD Rank Selection. | Registered as 'mgie' and 'mgie_svd'.
class MGIECompressor(BaseCompressor)
  doc: MGIE-SVD Compressor (Maximum Information Entropy SVD).
__init__(model, rank, max_error, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, min_rank, target_modules)
features: entropy,gqa,lora,lowrank,peft,rank,svd
loc: 342