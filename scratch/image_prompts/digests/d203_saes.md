# d203 saes  [compressor]
aliases: saes,saes_svd_v2
file: Codes/framework/compression_engine/saes_svd_v2.py  doc: Docs/content/compressions/saes_svd_v2.md
doc: SAES-SVD v2: Spectrum-Aware Error-Suppressed Singular Value Decomposition. | Registered in COMPRESSOR_REGISTRY as 'saes_svd_v2' and 'saes'. | Mathematical Formulation: | 1. Truncated SVD Baseline:
class SAESSVDCompressorV2(BaseCompressor)
  doc: SAES-SVD v2 Compressor with iterative spectrum-aware residual correction, | Frobenius validation, and safe fallback.
__init__(model, rank, rank_ratio, correcting_steps, target_modules, target_attention_only, max_error, fallback_to_original)
features: gating,lora,lowrank,peft,rank,svd
loc: 305