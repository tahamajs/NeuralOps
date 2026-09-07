# d191 mlorq  [compressor]
aliases: mlorq,mlorq_compressor
file: Codes/framework/compression_engine/mlorq_compressor.py  doc: Docs/content/compressions/mlorq_compressor.md
doc: MLoRQ: Multi-Layer Optimal Rank & Quantisation under a Total Memory Budget. | Registered as 'mlorq' and 'mlorq_compressor'.
class MLoRQLoRALinear(Module)
  doc: Linear layer containing MLoRQ low-rank quantized base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class MLoRQCompressor(BaseCompressor)
  doc: MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget. | Production hardening features:
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, total_memory_budget_mb, bits_candidates, max_error, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, min_rank, target_modules)
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: gqa,lora,peft,quant,rank,svd
loc: 474