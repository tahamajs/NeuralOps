# d202 rpca  [compressor]
aliases: rpca
file: Codes/framework/compression_engine/rpca_compressor.py  doc: NONE
doc: Robust PCA: decompose W = L + S, factorizing low-rank component L with structural SVDLinear. | Registered in COMPRESSOR_REGISTRY as 'rpca', 'rpca_factory', and 'rpca_compressor_factory'.
class RPCACompressor(BaseCompressor)
  doc: Robust Principal Component Analysis (RPCA) Low-Rank Compressor. | Decomposes weight matrices W into low-rank component L and sparse outlier matrix S
__init__(model, rank, mu, max_iter, min_rank, skip_attention_qkv, target_attention_only, use_lora, lora_r, lora_alpha, target_modules, max_error)
tensor_ops:
  W = raw_w.t().contiguous().float()
  W = raw_w.contiguous().float()
  W_rec = torch.matmul(U_final.float(), V_final.float())
features: gqa,lora,lowrank,peft,pool,rank,sparsity,svd
loc: 412