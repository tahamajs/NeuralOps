# d172 flat_llm  [compressor]
aliases: flat_llm,flat_llm_compressor
file: Codes/framework/compression_engine/flat_llm.py  doc: Docs/content/compressions/flat_llm.md
doc: FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning via Eigen-Decomposition. | Registered in COMPRESSOR_REGISTRY as 'flat_llm' and 'flat_llm_compressor'. Production hardened | for multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with reconstruction | validation, Conv1D support, and safe fallback.
class FLATLLMCompressor(BaseCompressor)
  doc: FLAT-LLM Compressor: Activation-aware neuron and output channel pruning. | Production hardening features:
__init__(model, tokenizer, keep_ratio, damping, max_error, target_attention_only, fallback_to_original, calibration_loader, calibration_texts, device, structured_24, target_modules)
tensor_ops:
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
  H = torch.matmul(act_flat.T, act_flat) / max(act_flat.shape[0], 1)
features: calib,distill,gating,lora,peft,pool,sparsity,topk
loc: 394