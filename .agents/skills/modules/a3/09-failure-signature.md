# Failure Signature — a3

## Known Empirical Failures

### Failure 1: Parameter explosion (+18% to +21%) instead of compression

**Symptom:** Observed in benchmark results.
**Root cause:** In benchmark runs without calibration data, fallback_to_original triggered for layers exceeding max_error (0.40) while LoRA adapters were still added, resulting in net parameter growth.
**Fix:** Must supply calibration loader with >= 128 samples to enable true factorization and set max_error higher or fallback_to_original=False.


