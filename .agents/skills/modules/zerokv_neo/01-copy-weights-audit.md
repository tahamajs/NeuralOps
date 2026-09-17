# Copy Weights Audit — zerokv_neo

## Purpose

Verify that weight transfer actually transfers pretrained weights.
Implementation details from `Codes/framework/compression_engine/zerokv_neo_compressor.py`:

In-place KV cache pruner. Does not replace weight tensors. Checks _is_attention_proj to wrap layers with LoRA adapters while leaving weights untouched.

## Verification criteria

- Verify weight Frobenius norm ratio > 5.0 relative to random initialization.
- For factorized projections, verify reconstruction error against original dense weights.

## Output

Logged to `benchmark_logs/unified_zerokv_neo_*.log`.
