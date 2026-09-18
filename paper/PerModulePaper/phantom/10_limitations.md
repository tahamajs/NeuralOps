<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 (classification) is evaluated. PHANTOM may behave differently on:
- Generation tasks (summarization, translation)
- Long-context tasks (>512 tokens)
- Retrieval (dense passage)
- Multi-class classification

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |

## 10.3 Hardware

All experiments on RTX 4070 Laptop. Not tested on:
- A100/H100
- Edge devices
- CPU inference

## 10.4 Multi-Round LSH Overhead

4 rounds × attention = 4× attention compute. For short sequences 
($L < 32$), this may dominate.

## 10.5 Bucket Imbalance

LSH buckets can be imbalanced. Some buckets may contain many tokens, 
negating sparsity benefits.

## 10.6 Softpick Fallback

The softmax fallback activates on ~0.85% of rows, losing 
sparsity for those rows.

## 10.7 Single-Seed Results

All main results from seed 42. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.8 No Formal Downstream Bound

We provide reconstruction bounds for CUR (Prop 4.7.1), but no downstream 
task bound.

## 10.9 CUR Failure Cases

For rank-deficient or ill-conditioned matrices, pseudoinverse may be 
numerically unstable. Fallback to Kaiming init is possible but rare.

## 10.10 No Comparison with Fixed-Pattern Sparse

We do not directly compare with Longformer or BigBird (different training 
paradigms).

## 10.11 Failure Modes Not Fully Diagnosed

0.85

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No unit tests for edge cases
- Numerical stability could be improved

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.

---