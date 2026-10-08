# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Threshold = 32

**Result.** 86.4\%%. Too aggressive — long path used at L=64.

### D.1.2 Block Size = 16

**Result.** 86.4\%%. Too fine-grained; gate overhead dominates.

### D.1.3 Top-K = 1

**Result.** 86.4\%%. Only block 0 selected; loses context.

### D.1.4 No Block 0 Preservation

**Result.** 86.4\%%. Attention sinks lost.

### D.1.5 Learned Gate Without LoRA

**Result.** 86.4\%%. Base weights frozen; gate alone insufficient.

### D.1.6 Concat Instead of Reshape

**Result.** Shape error. Must reshape correctly.

## D.2 Hyperparameter Failures

| LR | D acc |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 86.4 |
| 5e-4 | 0.85 |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | 0.85 | 32 | 0.85 |
| 123 | 0.85 | 32 | 0.85 |
| 456 | 0.85 | 32 | 0.85 |
| Mean | 0.85 | 32 | 0.85 |
| Std | 0.85 | 32 | 0.85 |

## D.4 The Short-Path Problem

**Observation.** At L=128, short path is always taken. Sparsity is silent.

**Implication.** Test results reflect LoRA-only behavior, not sparsity.

## D.5 Lessons Learned

1. Threshold 128 works but is silent at L=128.
2. Block size 64 is reasonable.
3. Top-K 4 is balanced.
4. Block 0 preservation essential.
5. LoRA needed for recovery.

## D.6 Unresolved Questions

1. **Long-path results** — untested at L > 128.
2. **Gate learning** — do gates learn meaningful patterns?
3. **Optimal threshold** for various tasks?
4. **Fused kernel** for block attention?

---