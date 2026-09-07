# d184 kvtc  [compressor]
aliases: kvtc,kvtc_quantizer
file: Codes/framework/compression_engine/kvtc_compressor.py  doc: NONE
doc: KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive quantization | + entropy coding for compact KV cache storage. | Paper: "KV Cache Transform Coding for Compact Storage in LLM Inference" | Staniszewski & Lancucki, ICLR 2026.
class SimpleEntropyCoder()
  doc: Lightweight entropy coder using empirical symbol frequency estimation.
class KVTCStore()
  doc: Runtime KV Cache Transform Coding engine. | Applies PCA decorrelation across head feature dimensions and adaptive quantization,
class KVTCQuantizedLoRALinear(Module)
  doc: Linear layer containing KVTC quantized low-rank base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class KVTCCompressor(BaseCompressor)
  doc: KVTC: KV Cache Transform Coding compressor. | Compresses linear projection weights using PCA decorrelation,
__init__(bits, pca_energy)
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, bits, pca_energy, max_error, fallback_to_original, lora_r, lora_alpha, target_attention_only, skip_attention_qkv, min_rank, target_modules)
tensor_ops:
  k_flat = k.detach().float().reshape(-1, k.shape[-1])
  v_flat = v.detach().float().reshape(-1, v.shape[-1])
  k_flat = k.reshape(-1, D)
  v_flat = v.reshape(-1, D)
  k_trans = torch.matmul(k_flat, self.pca_basis_k)
  v_trans = torch.matmul(v_flat, self.pca_basis_v)
  k_hat = torch.matmul(q_k * scale_k, self.pca_basis_k.t()).reshape(orig_shape)
  v_hat = torch.matmul(q_v * scale_v, self.pca_basis_v.t()).reshape(orig_shape)
shapes: [B, H, N, D
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: adaptive,calib,entropy,gating,gqa,lora,lowrank,peft,pool,quant,rank,svd
loc: 560