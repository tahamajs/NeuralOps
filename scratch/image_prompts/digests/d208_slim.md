# d208 slim  [compressor]
aliases: slim,slim_compressor
file: Codes/framework/compression_engine/slim_compressor.py  doc: Docs/content/compressions/slim_compressor.md
doc: SLiM: Structured Low-Rank Integrated Matrix Compression (Quantization + 2:4 Sparsity + SVD Residual Correction). | Registered in COMPRESSOR_REGISTRY as 'slim' and 'slim_compressor'. | Mathematical Formulation: | 1. Uniform/Non-Uniform b-bit Quantization:
class SLiMCompressor(BaseCompressor)
  doc: SLiM Compressor: One-shot joint b-bit quantization, 2:4 structured sparsity, and low-rank residual recovery.
__init__(model, bits, sparsity, rank, structured_24, target_attention_only, max_error, fallback_to_original, target_modules)
tensor_ops:
  flat = W_q.abs().reshape(-1)
  correction = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
features: gating,lora,lowrank,peft,quant,rank,sparsity,svd,topk
loc: 282