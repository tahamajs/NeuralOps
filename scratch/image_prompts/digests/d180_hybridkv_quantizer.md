# d180 hybridkv_quantizer  [compressor]
aliases: hybridkv_quantizer
file: Codes/framework/compression_engine/hybridkv_compressor.py  doc: NONE
doc: HybridKV: Hybrid KV Cache Compression for Multimodal & Long-Context LLMs. | Classifies attention heads into static/dynamic types and applies distinct | compression strategies per type under a hierarchical budget strictly to dynamic KV caches. | Paper: "HybridKV: Hybrid KV Cache Compression for Efficient MLLM Inference"
class HybridKVCache()
  doc: Dynamic KV Cache Manager for HybridKV. | Maintains separate retention policies for static vs dynamic attention heads:
class HybridKVCompressor(BaseCompressor)
  doc: HybridKV Compressor: Classifies projection heads into static/dynamic types | and attaches HybridKVCache instances to attention layers without modifying model weights.
__init__(max_cache_len, num_sinks, recent_window, budget, static_ratio, static_heads_mask)
__init__(model, budget, static_ratio, max_cache_len, num_sinks, recent_window, target_attention_only, cfg)
tensor_ops:
  self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
  self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
  if any(part in name.split(".") for part in ["lora_A", "lora_B"]):
shapes: [B, H, N, D [B, N, D [N, D
features: chunked,gating,lora,peft,pool,topk,window
loc: 340