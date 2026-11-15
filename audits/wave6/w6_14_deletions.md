# Deletion Report: Example 20

- **File:** `examples/20_long_context_niah_eval.py`
- **Verdict:** DELETED
- **Reason:** Completely fabricated NIAH evaluation using `torch.randn` for attention matrices and simulated H2O eviction over random tensors, with a hardcoded `pass_fail(..., True)`. Authentic NIAH requires 8k+ real tokens and an authentic LM checkpoint prohibitive for lightweight CPU examples.
- **Replacement:** Use `neuralops.evaluation.niah_passkey` and benchmark suite `neuralops.benchmarks.suites.niah`. Full rationale documented in `audits/wave6/w6_14_verdict.md`.
