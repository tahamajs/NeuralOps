# Copy Weights Audit — phantom

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/attentions/phantom.py`:

Uses statistical leverage scores from SVD to sample rows and columns for CUR matrix decomposition (C @ U_pinv @ R). Transfers pretrained weights.

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_phantom_*.log`.
