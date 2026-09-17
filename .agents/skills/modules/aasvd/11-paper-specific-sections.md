# Paper Specific Sections — aasvd

## Methodology Description

AASVD: Activation-Aware SVD Compressor with Streaming EMA Covariance
Source implementation: `Codes/framework/compression_engine/aa_svd_compressor.py`.

Mathematical formulation:
$$
C_X^{(t)} = \beta C_X^{(t-1)} + (1 - \beta) \frac{1}{B} X^\top X, \quad y = (x V^\top) U^\top + \frac{\alpha}{r} (x A^\top) B^\top
$$

## Results Narrative

- DistilBERT: 87.96% accuracy (PASS), -2.42% parameter change.
- RoBERTa: 86.7% accuracy (PASS), -1.18% parameter change.
- GPT-2: 69.38% accuracy (PASS), -9.95% parameter change.
