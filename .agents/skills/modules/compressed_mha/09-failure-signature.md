# Failure Signature — compressed_mha

## Known Empirical Failures

### Failure 1: Universal 50.92% collapse across all backbones

**Symptom:** Observed in benchmark results.
**Root cause:** Aggressive low-rank truncation (rank 64 on 768-dim QKV space) without activation weighting strips critical high-rank attention channels. 100 recovery steps are insufficient to restore representations.
**Fix:** Increase default rank to 192 or use activation-aware preconditioning before SVD factorization.


