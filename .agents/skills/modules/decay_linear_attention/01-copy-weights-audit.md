# Copy Weights Audit — decay_linear_attention

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/attentions/decay_linear_attention.py`:

Transfers pretrained Q, K, V and Out projection weights into LoRALinear wrappers with Frobenius norm verification (>= 1e-4).

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_decay_linear_attention_*.log`.
