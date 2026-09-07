# d187 lsc  [compressor]
aliases: lsc,lsc_compressor
file: Codes/framework/compression_engine/lsc_compressor.py  doc: Docs/content/compressions/lsc_compressor.md
doc: LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor. | Registered in COMPRESSOR_REGISTRY as 'lsc' and 'lsc_compressor'. | Mathematical Formulation: | 1. 2D Discrete Fourier Transform:
class LSCPrunedLoRALinear(Module)
  doc: Linear layer containing 2D Fourier spectral filtered base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class LSCCompressor(BaseCompressor)
  doc: LSC Compressor: 2D Fourier spectral low-rank and frequency filtering compressor. | Production hardening features:
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, rank, energy_ratio, max_error, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, target_modules)
tensor_ops:
  flat_mag = magnitude.reshape(-1)
  flat_mag = magnitude.reshape(-1)
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: gqa,lora,peft,rank,topk
loc: 440