# d147 admm  [compressor]
aliases: admm,admm_compressor
file: Codes/framework/compression_engine/admm_compressor.py  doc: Docs/content/compressions/admm_compressor.md
doc: ADMM Compressor: Alternating Direction Method of Multipliers for Joint Pruning and Quantization. | Registered in COMPRESSOR_REGISTRY as 'admm' and 'admm_compressor'. Production hardened with | reconstruction validation, Conv1D support, FP32 numerical stability, safe fallback, | LoRALinear preservation, and attention-only targeting.
class ADMMCompressor(BaseCompressor)
  doc: ADMM-based Joint Structured Pruning and Non-Uniform / Uniform Quantization Engine. | Production hardening features:
__init__(model, sparsity, quant_bits, rho1, rho2, max_iter, structured_24, block_size, max_error, fallback_to_original, target_attention_only, target_modules)
tensor_ops:
  flat = tensor.abs().reshape(-1)
features: gating,lora,peft,pool,quant,sparsity,ssm,topk
loc: 416