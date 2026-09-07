# d220 zerokv  [compressor]
aliases: zerokv,zerokv_neo,zerokv_neo_compressor
file: Codes/framework/compression_engine/zerokv_neo_compressor.py  doc: Docs/content/compressions/zerokv_neo_compressor.md
doc: ZeroKV-Neo: L2-norm based KV cache eviction with attention sink and heavy-hitter preservation. | Registered as 'zerokv_neo'. | Production-hardened with: | - Bounded O(1) cache memory management with safe dimension handling
class ZeroKVNeoCache()
  doc: Dynamic KV Cache Manager for ZeroKV-Neo. | Maintains bounded O(1) cache memory by partitioning budget:
class ZeroKVNeoCompressor(BaseCompressor)
  doc: ZeroKV-Neo Compressor: L2-norm weight pruning preserving attention sinks and heavy hitters. | Args:
__init__(max_cache_len, num_attention_sinks, recent_window)
__init__(model, keep_ratio, num_attention_sinks, heavy_hitter_ratio, max_error, max_retries, fallback_to_original, target_attention_only)
tensor_ops:
  self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
  self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
  idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
  self.k_cache = torch.cat([sink_k, sel_mid_k, win_k], dim=2).contiguous()
  self.v_cache = torch.cat([sink_v, sel_mid_v, win_v], dim=2).contiguous()
  self.k_cache = torch.cat([sink_k, win_k], dim=2).contiguous() if window > 0 else sink_k
  self.v_cache = torch.cat([sink_v, win_v], dim=2).contiguous() if window > 0 else sink_v
  W = raw_w.t().float().contiguous()
shapes: [B, H, N, D
features: adaptive,lora,peft,pool,topk,window
loc: 428