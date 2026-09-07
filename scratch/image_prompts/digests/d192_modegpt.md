# d192 modegpt  [compressor]
aliases: modegpt
file: Codes/framework/compression_engine/modegpt_compressor.py  doc: NONE
doc: MoDeGPT: Modular Decomposition of Transformer Blocks. | Decomposes pairs of linear projections (e.g. Q & K, Up & Gate) jointly via shared low-rank bases. | Registered as 'modegpt'.
class MoDeGPTLoRALinear(Module)
  doc: Linear layer containing MoDeGPT decomposed base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class MoDeGPTCompressor(BaseCompressor)
  doc: MoDeGPT: Modular Decomposition of Transformer Projection Blocks. | Decomposes paired linear projections (e.g. Q & K, Gate & Up) jointly into shared low-rank subspaces:
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, rank, method, max_error, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, min_rank, target_modules)
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: gating,gqa,lora,peft,rank,svd
loc: 553