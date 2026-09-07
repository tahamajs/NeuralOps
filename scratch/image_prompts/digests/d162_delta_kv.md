# d162 delta_kv  [compressor]
aliases: delta_kv,delta_kv_compressor
file: Codes/framework/compression_engine/delta_kv.py  doc: Docs/content/compressions/delta_kv.md
doc: DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity. | Encodes semantic residuals relative to retrieved historical references, | preserving attention fidelity while drastically compressing dynamic KV caches. | Paper: "DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity"
class DeltaKVStore()
  doc: Maintains historical reference key-value representations and compresses incoming K/V tensors | by storing delta residuals relative to top-k closest reference tokens.
class DeltaKVCompressor(BaseCompressor)
  doc: DeltaKV Compressor: Decomposes projection weights into a low-rank reference component | plus sparse residuals, modeling weight matrices under residual delta formulation.
__init__(max_refs, topk, compression_ratio, residual_threshold)
__init__(model, rank, sparsity, max_error, target_attention_only, fallback_to_original, cfg)
tensor_ops:
  self.ref_k = self.ref_k[:1].expand(B, -1, -1, -1).clone()
  self.ref_v = self.ref_v[:1].expand(B, -1, -1, -1).clone()
  self.ref_k = self.ref_k[:, :, -self.max_refs :, :].contiguous()
  self.ref_v = self.ref_v[:, :, -self.max_refs :, :].contiguous()
  ref_k_b = ref_k_cur if ref_b == B else ref_k_cur[:1].expand(B, -1, -1, -1)
  ref_v_b = ref_v_cur if ref_b == B else ref_v_cur[:1].expand(B, -1, -1, -1)
  sim = torch.matmul(k_4d.float(), ref_k_b.float().transpose(-2, -1)) / (D**0.5)
  top_idx_exp = top_idx.unsqueeze(-1).expand(B, H, N, tk, D)
shapes: [B, H, N, D [B, H, N, tk [B, H, N, tk, D [B, N, D [N, D
features: gating,lora,peft,pool,rank,sparsity,svd,topk
loc: 414