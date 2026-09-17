# Adaptive SVD Compressor with Spectral Importance Allocation (adasvd_compressor)

**Registry Keys**: `adasvd` <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 41 -->
**Source**: `Codes/framework/compression_engine/adasvd_compressor.py` (488 lines)
**Type**: Spectral Nuclear-Norm Guided SVD with Alternating Least Squares Refinement
**Base Class**: `AdaSVDCompressor(BaseCompressor)` <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 42 -->

## Abstract
`adasvd_compressor` dynamically allocates low-rank factorization ranks across transformer projection layers proportionally to spectral importance (nuclear norm $\|W\|_* = \sum_i \sigma_i$). It distributes singular values symmetrically ($A = U \sqrt{\Sigma}, B = \sqrt{\Sigma} V^	op$), applies iterative Alternating Least Squares (ALS) factor refinement, and replaces target dense projections with factorized `SVDLinear` modules. Across DistilBERT, RoBERTa, and GPT-2, it achieves verified parameter compression (up to 12.0% parameter reduction on RoBERTa) while maintaining high classification fidelity.
