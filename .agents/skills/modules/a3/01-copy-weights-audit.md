# Copy Weights Audit — a3

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/compression_engine/a3_compressor.py`:

Unweights SVD factors U_r, V_r using 1 / sqrt(F + eps). Preserves linear module bias and instantiates A3FactorizedLinear.

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_a3_*.log`.
