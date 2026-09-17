# Activation-Aware Singular Value Decomposition Compressor (aa_svd_compressor)

**Registry Keys**: `aasvd`, `aa_svd`, `aa_svd_compressor`, `aa_svd_factory`
**Source**: `Codes/framework/compression_engine/aa_svd_compressor.py` <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py lines 202-205 -->
**Type**: Structural Linear Compression via Low-Rank Factorization ($y = (x V^\top) U^\top$)
**Base Class**: `AASVDCompressor(BaseCompressor)` <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 205 -->

## Abstract
Standard Singular Value Decomposition (SVD) truncates weight matrices based solely on static parameter magnitudes, treating all spectral directions equally regardless of inference input distributions. This leads to catastrophic representation drift in transformer attention projections where hidden activations exhibit severe outlier dimensions. We present **Activation-Aware SVD (AASVD)** (`aa_svd_compressor`), a production-hardened compression engine that collects streaming Exponential Moving Average (EMA) activation covariances $C_X$, preconditions weights via symmetric whitening $M = W C_X^{1/2}$, and replaces dense linear layers with physical low-rank factorized `SVDLinear` modules.
