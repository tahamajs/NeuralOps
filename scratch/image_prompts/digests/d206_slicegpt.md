# d206 slicegpt  [compressor]
aliases: slicegpt,slicegpt_compressor
file: Codes/framework/compression_engine/slicegpt.py  doc: Docs/content/compressions/slicegpt.md
doc: SliceGPT: Compress Large Language Models by Deleting Rows and Columns. | Registered in COMPRESSOR_REGISTRY as 'slicegpt'. | Mathematical Formulation: | 1. Orthogonal Coordinate Invariance:
class SliceGPTCompressor(BaseCompressor)
  doc: SliceGPT Compression Strategy: Projects activations onto principal components via orthogonal | rotation matrices Q and slices out lowest-variance coordinates, with Frobenius validation and fallback.
__init__(model, tokenizer, sparsity, non_uniform, calibration_loader, calibration_texts, target_attention_only, device, max_error, fallback_to_original, target_modules)
tensor_ops:
  X_flat = X.reshape(-1, X.shape[-1]).float().to(device)
  cov = torch.matmul(X_flat.T, X_flat) / X_flat.shape[0]
  W_tilde = torch.matmul(W, Q)
  W_pruned = torch.matmul(W_sliced_rot, Q.T)
formulas/comments:
  Rotate weight: W_tilde = W @ Q
  Rotate back to original coordinate basis: W_pruned = W_sliced_rot @ Q^T
features: calib,gating,lora,peft,rotation,sparsity
loc: 342