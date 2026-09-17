# Copy Weights Audit — compressed_mha

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/attentions/compressed_mha_attention.py`:

Stacks pretrained Q, K, V weights vertically into W_qkv, runs torch.linalg.svd, and truncates to rank_qkv. Pretrained output projection W_o is truncated to rank_out. Check norm ratio > 5.

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_compressed_mha_*.log`.
