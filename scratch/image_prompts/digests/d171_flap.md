# d171 flap  [compressor]
aliases: flap,flap_compressor
file: Codes/framework/compression_engine/flap.py  doc: Docs/content/compressions/flap.md
doc: FLAP: Fluctuation-Based Adaptive Structured Pruning. | Registered in COMPRESSOR_REGISTRY as 'flap' and 'flap_compressor'. Production hardened | for multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with | reconstruction validation, Conv1D support, and safe fallback.
class FLAPCompressor(BaseCompressor)
  doc: FLAP Compressor: Fluctuation-based Adaptive Structured Pruning. | Production hardening features:
__init__(model, tokenizer, sparsity, metric_type, max_error, target_attention_only, fallback_to_original, calibration_loader, calibration_texts, device, structured_24, target_modules)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  flat = inp.reshape(-1, in_f)
  X = activations.reshape(-1, activations.shape[-1]).float().to(device)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
features: adaptive,calib,distill,gating,lora,peft,pool,sparsity,topk
loc: 416