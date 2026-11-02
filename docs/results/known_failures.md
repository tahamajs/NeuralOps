# Known Failure Modes and Theoretical Failure Boundaries

This document catalogs mathematically expected and empirically documented failure modes across compression and kernel execution regimes in NeuralOps.

---

## 1. Direct SVD Rank Collapse ($r < 8$, $r = 4$)

- **Category**: Low-Rank Matrix Factorization / Unwhitened SVD
- **Affected Compressor**: `DirectSVDCompressor` (`neuralops.compression.svd.direct_svd`)
- **Empirical Findings**:
  - **Downstream Accuracy Collapse**: Direct SVD rank 4 yields accuracy 34.8% on DistilBERT and 36.2% on RoBERTa on classification tasks, which is worse than random chance.
  - **Perplexity Explosion**: Truncating transformer weight matrices to rank 4 causes language modeling perplexity to explode to $PPL > 1000$ on WikiText-2 ($PPL > 1200$).
- **Theoretical Root Cause**: Truncating 768-dimensional transformer layers to rank 4 discards $> 88\%$ of spectral energy (retains $< 4\%$ spectral energy). Without activation covariance whitening (or HNC-SVD pullback continuation), the unweighted singular vectors fail to preserve the manifold geometry where representations lie.
- **Safety Mechanisms & Invariants**:
  - `DirectSVDCompressor` raises `ValueError` if `rank < 4`:
    `"rank={rank} retains <4% spectral energy on 768-dim transformer layers; accuracy collapse is mathematically guaranteed. Use rank >= 16 or Whitened SVD."`
  - `DirectSVDCompressor` logs a warning if `rank < 8`:
    `"rank={rank} retains <15% spectral energy. Expect significant accuracy loss. Prefer rank >= 16 or Whitened SVD."`
- **Theoretical Justification**: These failure boundaries are expected theoretical properties of unwhitened spectral truncation, proving the mathematical necessity of Riemannian/Hessian nullspace continuation (HNC-SVD) and activation whitening.

---

## 2. Uncalibrated Covariance Estimation in Adaptive Compressors

- **Category**: Activation-Aware Low-Rank Decomposition
- **Affected Compressors**: `AASVDCompressor` (`aa_svd_factory`), `AdaSVDCompressor` (`adasvd_compressor`)
- **Empirical Findings**:
  - Running without activation covariance calibration data causes adaptive rank allocation to assign rank 0 or degenerated projections to critical layers, producing incoherent outputs ($PPL = 1046.66$ for `aa_svd_factory`, $PPL = 468.30$ for `adasvd_compressor`).
- **Theoretical Root Cause**: Accurate Fisher or covariance estimation requires forward-pass activation sampling. Without real calibration data, uncalibrated approximations fail to preserve high-energy activation subspaces.
- **Safety Mechanisms & Invariants**:
  - `AdaSVDCompressor` sets `requires_calibration = True`.
  - When invoked without calibration data, `compress()` fails fast by raising:
    `ValueError: AdaSVDCompressor requires calibration data for covariance estimation. Provide calibration_data=torch.Tensor or use a compressor that does not require calibration (e.g., direct_svd with rank >= 16).`
