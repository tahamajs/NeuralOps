# d153 cachenotes  [compressor]
aliases: cachenotes,cachenotes_compressor
file: Codes/framework/compression_engine/cachenotes_compressor.py  doc: Docs/content/compressions/cachenotes_compressor.md
doc: CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression | with Learnable Continuous CPT (Compression-Planning-Token) Generator. | Given a task description, CacheNotes generates Compression-Planning-Tokens (CPTs) | that guide the compression of the corpus and KV cache into compact, reusable states.
class CacheNotesLoRALinear(Module)
  doc: Linear layer containing CacheNotes compressed base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class CompressionPlanningTokens()
  doc: Static CPT heuristic importance generator. | Serves as safe baseline fallback when learnable CPT generator is inactive.
class CPTGenerator(Module)
  doc: Learnable Task-Aware Compression Planning Token (CPT) Generator. | Maps arbitrary task descriptions to continuous per-feature importance vectors for weight SVD
class CacheNotesKVCache()
  doc: Runtime task-guided dynamic KV cache compressor. | Evaluates token task-relevance dynamically and prunes non-essential context tokens,
class CacheNotesCompressor(BaseCompressor)
  doc: CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning | with continuous learnable CPT Generator.
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(hidden_dim, num_freqs, vocab_size)
forward(task, d, device)
__init__(max_cache_len, task_type, keep_ratio, num_sinks, cpt_generator)
__init__(model, task_type, task_description, use_learned_cpt, keep_ratio, max_error, fallback_to_original, protect_critical, target_attention_only, lora_r, lora_alpha, skip_attention_qkv)
tensor_ops:
  x_flat = x.reshape(-1, self.in_features)
  return out.reshape(*orig_shape[:-1], self.out_features)
  words = str(text).lower().replace(",", " ").replace(".", " ").replace(":", " ").replace("-", " ").split()
  h_exp = h.unsqueeze(0).expand(d, -1)  # [d, hidden_dim]
  self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
  self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
  weighted_k = self.k_cache * cpt_importance.view(1, 1, 1, D)
  idx_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
shapes: [1, 1, 1, D [B, H, N, D
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
formulas/comments:
  Pure static compression (lora_r == 0)
features: calib,gating,gqa,hash,lora,lowrank,peft,pool,rank,svd,topk
loc: 896