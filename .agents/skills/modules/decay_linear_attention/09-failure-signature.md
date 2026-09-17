# Failure Signature — decay_linear_attention

## Known Empirical Failures

### Failure 1: Model size grows (LoRA parameters added to non-factorized weights)

**Symptom:** Observed in benchmark results.
**Root cause:** Decay linear attention replaces quadratic softmax attention with linear recurrent attention for FLOP/throughput efficiency, not parameter reduction. Adding LoRA adapters makes parameter change positive.
**Fix:** Honest reporting: classify as Linear Recurrent Attention / Throughput Acceleration, NOT Parameter Compression.


