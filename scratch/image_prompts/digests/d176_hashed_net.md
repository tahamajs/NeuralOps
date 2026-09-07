# d176 hashed_net  [compressor]
aliases: hashed_net,hashed_net_compressor,hashednets,hashednets_compressor
file: Codes/framework/compression_engine/hashed_net_compressor.py  doc: Docs/content/compressions/hashed_net_compressor.md
doc: HashedNet Compressor: replaces nn.Linear and Conv1D layers with HashedLinear. | Registered in COMPRESSOR_REGISTRY as 'hashed_net', 'hashednets', and 'hashednets_compressor'.
class HashedNetCompressor(BaseCompressor)
  doc: Compress a model by replacing target linear/Conv1D layers with HashedLinear, | which shares weights globally through compact hash tables.
__init__(model, n_buckets, compression_ratio, rank, max_error, target_attention_only, fallback_to_original, max_retries, use_sign, lora_rank, lora_alpha, freeze_hash_table)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
features: adaptive,gating,gqa,hash,lora,peft,pool,rank
loc: 417