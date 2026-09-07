# d178 hybrid_svd  [compressor]
aliases: direct_svd_encoder,hybrid_svd
file: Codes/framework/compression_engine/hybrid_svd_layer.py  doc: NONE
doc: Hybrid SVD Layer: single-head SVD attention replacement with positional encoding, | plus full encoder layer and compression engine. | Registered as 'hybrid_svd' and 'direct_svd_encoder'.
class SVDSingleHead(Module)
  doc: Single-head SVD attention approximation with sinusoidal positional encoding.
class DirectSVDEncoderLayer(Module)
  doc: Encoder layer that replaces multi-head attention with SVDSingleHead + standard FFN.
class HybridSVDCompressor(BaseCompressor)
  doc: Compressor supporting both linear-level SVD replacement and architecture-safe attention compression.
__init__(hidden_size, svd_rank)
forward(hidden_states)
__init__(config, hidden_size, intermediate_size, num_attention_heads, svd_rank, layer_norm_eps, hidden_dropout_prob, lora_r, lora_alpha)
forward(hidden_states, attention_mask, head_mask, output_attentions)
__init__(model, svd_rank, use_full_layer_replacement, lora_r, lora_alpha, max_error, target_attention_only, fallback_to_original, cfg, target_modules)
tensor_ops:
  x_flat = F.normalize(x.reshape(-1, self.hidden_size).float(), p=2, dim=-1)
  weighted_U = U[:, :k] * S[:k].view(1, -1)
  output_flat = torch.matmul(weighted_U, Vh[:k, :])
  output = output_flat.reshape(batch_size, seq_len, self.hidden_size).to(dtype=hidden_states.dtype)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  W_recon = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
features: gating,lora,lowrank,peft,pool,rank,svd
loc: 440