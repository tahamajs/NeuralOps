# Failure Signature — zerokv_neo

## Known Empirical Failures

### Failure 1: 50.92% collapse on RoBERTa and GPT-2

**Symptom:** Observed in benchmark results.
**Root cause:** ZeroKV is an inference-time KV pruning compressor. When evaluated with 100 LoRA recovery steps on bidirectional encoders or causal decoders without calibration/fine-tuning matched to pruning masks, representations collapse to majority class.
**Fix:** Increase recovery steps or retain full attention during classification token positions.

### Failure 2: Zero parameter reduction (0.0%)

**Symptom:** Observed in benchmark results.
**Root cause:** ZeroKV prunes KV cache tensors dynamically at inference time; it does not truncate or factorize weight matrices. Weight parameter count remains exactly identical.
**Fix:** Honest reporting: classify as KV Cache Memory Reduction, NOT Weight Parameter Compression.


