# d166 enlighten  [compressor]
aliases: enlighten
file: Codes/framework/compression_engine/enlighten_compressor.py  doc: NONE
doc: ENLighten: Low-Rank SVD + 2:4 Structured Sparsity (Photonic-Aligned). | Registered as 'enlighten'. Production hardened for multi-architecture compatibility | (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with vectorization, error validation, and safe fallback.
class ENLightenCompressor(BaseCompressor)
  doc: ENLighten Compressor: Low-Rank SVD + 2:4 Structured Sparsity. | Decomposes weights into low-rank factors plus 2:4 structured sparse residuals,
__init__(model, rank, sparsity, structured_24, max_error, target_attention_only, fallback_to_original, max_retries, target_modules, cfg)
tensor_ops:
  blocks = res_padded.view(out_f, num_blocks, 4)
  pruned_padded = pruned_blocks.view(out_f, -1)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  if not any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
features: distill,gating,lora,peft,pool,rank,sparsity,svd,topk
loc: 378