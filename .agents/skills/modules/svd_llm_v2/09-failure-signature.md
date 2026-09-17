# Failure Signature — svd_llm_v2

## Known Empirical Failures

### Failure 1: RoBERTa & GPT-2 collapse to 50.92% at 18.5% compression

**Symptom:** Observed in benchmark results.
**Root cause:** Direct SVD truncation of QKV and Out projections without Fisher calibration disrupts LayerNorm output distributions. DistilBERT recovers (81.88%), but RoBERTa/GPT-2 require LayerNorm recalibration.
**Fix:** Apply LayerNorm recalibration after SVD truncation before running LoRA recovery.


