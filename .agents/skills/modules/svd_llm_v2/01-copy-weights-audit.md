# Copy Weights Audit — svd_llm_v2

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/attentions/svd_llm_v2_attention.py`:

Explicitly factorizes Q, K, V, and Out projections into U, S, Vt buffers. Transposes HuggingFace Conv1D weights before factorization.

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_svd_llm_v2_*.log`.
