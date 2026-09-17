# Failure Signature — phantom

## Known Empirical Failures

### Failure 1: LSH bucket hashing and Softpick collapse on RoBERTa and GPT-2 (50.92%)

**Symptom:** Observed in benchmark results.
**Root cause:** Random hyperplanes in LSH partition keys and queries suboptimally when token embeddings are not centered, causing empty attention buckets. Softpick rectification exacerbates sparsity.
**Fix:** Use temperature scaling >= 2.0 or increase LSH rounds from 4 to 8 during decoder evaluation.


