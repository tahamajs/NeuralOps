# Adaptive Ranks SVD Compressor (adaptive_ranks)

**Registry Keys**: `adaptive_ranks`, `adaptive_svd`, `adaptive_ranks_factory`, `adaptive_svd_factory`, `adaptive_compressor`, `adaptive`
**Source**: `Codes/framework/compression_engine/adaptive_ranks.py` <!-- source: Codes/framework/compression_engine/adaptive_ranks.py lines 53-55 -->
**Type**: Dynamic Singular Value Energy Spectrum Factorization ($U_r = U_{:, :r}\sqrt{\Sigma_r}, V_r = \sqrt{\Sigma_r} V_{:r, :}^\top C_X^{-1/2}$)
**Base Class**: `AdaptiveRanksCompressor(BaseCompressor)` <!-- source: Codes/framework/compression_engine/adaptive_ranks.py line 55 -->

## Abstract
`adaptive_ranks` implements singular-value energy spectrum thresholding ($	au = 0.92$) with empirical activation covariance whitening and symmetric $\sqrt{\Sigma}$ singular value distribution. Rather than imposing rigid rank configurations, `adaptive_ranks` solves for the minimum rank per projection that satisfies cumulative energy criteria while enforcing a safety floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{	ext{in}}, d_{	ext{out}}) floor)$. Evaluated across DistilBERT, RoBERTa, GPT-2, and Qwen 3.5, it delivers consistent compression ratios up to 13.2% parameter reduction with high downstream accuracy.
