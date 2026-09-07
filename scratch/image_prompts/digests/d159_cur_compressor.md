# d159 cur_compressor  [compressor]
aliases: cur_compressor
file: Codes/framework/compression_engine/cur_compressor.py  doc: Docs/content/compressions/cur_compressor.md
doc: CUR Compressor: replaces Linear and Conv1D layers with CUR decomposition. | Uses leverage score sampling or energy sampling for column/row selection. | Registered in COMPRESSOR_REGISTRY as 'cur' and 'cur_compressor'.
class CURCompressor(BaseCompressor)
  doc: Compressor that replaces eligible Linear and Conv1D layers with CURLinear. | Args:
__init__(model, rank, min_rank, method, deterministic, max_error, fallback_to_original, max_retries, lora_r, lora_alpha, freeze_skeletons, target_attention_only)
features: adaptive,gating,lora,peft,pool,rank
loc: 401