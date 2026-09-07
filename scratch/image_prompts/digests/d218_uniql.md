# d218 uniql  [compressor]
aliases: uniql
file: Codes/framework/compression_engine/uniql_compressor.py  doc: NONE
doc: UniQL: Joint low‑rank + quantisation using SVD and uniform quant. Registered as 'uniql'. | Production-hardened with: | - Relative Frobenius reconstruction error validation (err <= max_error) | - Adaptive bit and rank retries
class UniQLCompressor(BaseCompressor)
  doc: UniQL Compressor: Joint Truncated SVD and Uniform Min-Max Quantization. | Args:
__init__(model, rank, bits, max_error, max_retries, fallback_to_original, target_attention_only)
tensor_ops:
  W = raw_w.t().float().contiguous()
  W = raw_w.float().contiguous()
  w_to_assign = best_w_rec.t().contiguous().to(dtype=orig_dtype, device=orig_device)
features: adaptive,lora,peft,pool,quant,rank,svd
loc: 335