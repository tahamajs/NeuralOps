# d160 cur_moe  [compressor]
aliases: cur_moe
file: Codes/framework/compression_engine/cur_moe_compressor.py  doc: NONE
doc: CUR-MoE Compressor: apply CUR cross-approximation to Mixture of Experts (MoE) modules and expert layers. | Registered in COMPRESSOR_REGISTRY as 'cur_moe' and 'cur_moe_compressor'.
class CURMoECompressor(BaseCompressor)
  doc: Compresses Mixture of Experts (MoE) layers and dense FFN layers using CUR cross-approximation. | Args:
__init__(model, rank, min_rank, method, deterministic, max_error, fallback_to_original, max_retries, target_attention_only, cfg)
tensor_ops:
  last_part = name.split(".")[-1]
features: adaptive,gating,lora,moe,peft,pool,rank
loc: 349