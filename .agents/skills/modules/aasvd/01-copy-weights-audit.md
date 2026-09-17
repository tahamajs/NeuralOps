# Copy Weights Audit — aasvd

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/compression_engine/aa_svd_compressor.py`:

Projects weight matrix onto principal covariance eigenvectors, sets U and V factors in SVDLinear, initializes LoRA adapter A and B.

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_aasvd_*.log`.
