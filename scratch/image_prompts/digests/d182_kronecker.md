# d182 kronecker  [compressor]
aliases: kronecker,kronecker_compressor
file: Codes/framework/compression_engine/kronecker_compressor.py  doc: Docs/content/compressions/kronecker_compressor.md
doc: Kronecker Product Compressor: replaces nn.Linear and Conv1D layers with KroneckerLinear. | Registered in COMPRESSOR_REGISTRY as 'kronecker'. Production hardened with | reconstruction validation, Conv1D support, adaptive SVD fallback, and critical layer protection.
class KroneckerCompressor(BaseCompressor)
  doc: Compresses a model by replacing eligible nn.Linear and Conv1D layers with a | Kronecker product decomposition (A (x) B), falling back to adaptive low-rank SVD
__init__(model, rank, max_error, fallback_to_original, lora_r, lora_alpha, target_attention_only, skip_attention_qkv, cfg, target_layers)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
features: adaptive,gating,gqa,kronecker,lora,peft,pool,rank,svd
loc: 346