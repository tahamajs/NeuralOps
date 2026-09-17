# Failure Signature — aasvd

## Known Empirical Failures

### Failure 1: Low compression ratio (1.18% - 9.95%)

**Symptom:** Observed in benchmark results.
**Root cause:** Rank 288 is conservative to guarantee high accuracy retention (86.7% RoBERTa, 87.96% DistilBERT). To achieve 20%+ compression, rank must be lowered to 128-192.
**Fix:** Tune rank parameter according to target compression threshold.


