# d210 sparsegpt  [compressor]
aliases: sparsegpt
file: Codes/framework/compression_engine/sparsegpt.py  doc: Docs/content/compressions/sparsegpt.md
doc: SparseGPT: One-Shot Layer-Wise Weight Pruning with Second-Order Error Compensation. | Registered in COMPRESSOR_REGISTRY as 'sparsegpt'. | Mathematical Formulation: | 1. Empirical Hessian Formulation:
class SparseGPTCompressor(BaseCompressor)
  doc: In-memory PyTorch SparseGPT Compressor. | Performs layer-by-layer second-order pruning with Optimal Brain Surgeon (OBS) weight
__init__(model, tokenizer, sparsity, prunen, prunem, block_size, percdamp, calibration_loader, calibration_texts, target_attention_only, device, structured_24)
tensor_ops:
  act_flat = activations.reshape(-1, activations.shape[-1]).float()
  update = torch.matmul(err_block, torch.matmul(inv_diag_block, h_cross))
features: calib,gating,lora,peft,pool,sparsity,topk
loc: 367