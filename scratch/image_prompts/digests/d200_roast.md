# d200 roast  [compressor]
aliases: roast,roast_compressor
file: Codes/framework/compression_engine/roast_hashed_linear.py  doc: NONE
doc: ROAST: Hardware-Aware Weight Sharing with Tiling. | Reference: 'ROAST: A Hardware-Aware Weight Sharing Scheme for Efficient Neural Networks' | Registered in COMPRESSOR_REGISTRY as 'roast', 'roast_compressor', and 'roast_factory'.
class ROASTLinear(Module)
  doc: Linear layer with weight sharing based on tiles (blocks) rather than | individual scalar weights. Each tile of size (tile_size, tile_size) shares
class ROASTCompressor(BaseCompressor)
  doc: Replace eligible linear and Conv1D layers with ROASTLinear (hardware-aware weight sharing).
__init__(in_features, out_features, n_buckets, tile_size, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, n_buckets, tile_size, skip_attention_qkv, use_lora, lora_r, lora_alpha, target_modules, target_attention_only, max_error, fallback_to_original, min_numel)
tensor_ops:
  tile_idx_i = torch.arange(self.tiles_out, device=device).unsqueeze(1).expand(-1, self.tiles_in)
  tile_idx_j = torch.arange(self.tiles_in, device=device).unsqueeze(0).expand(self.tiles_out, -1)
  W = raw_w.t().contiguous().float()
  W = raw_w.contiguous().float()
  W_orig = raw_w.t().contiguous().float() if is_conv1d else raw_w.contiguous().float()
parameters: torch.randn(self.n_buckets, device=device, dtype=dtype | torch.zeros(out_features, device=device, dtype=dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: gqa,hash,lora,peft,pool,rope
loc: 442