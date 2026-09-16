# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Threshold = 32

**Result.** {{THRESH32_ACC}}%. Too aggressive — long path used at L=64.

### D.1.2 Block Size = 16

**Result.** {{B16_ACC}}%. Too fine-grained; gate overhead dominates.

### D.1.3 Top-K = 1

**Result.** {{K1_ACC}}%. Only block 0 selected; loses context.

### D.1.4 No Block 0 Preservation

**Result.** {{NOBLOCK0_ACC}}%. Attention sinks lost.

### D.1.5 Learned Gate Without LoRA

**Result.** {{NO_LORA_ACC}}%. Base weights frozen; gate alone insufficient.

### D.1.6 Concat Instead of Reshape

**Result.** Shape error. Must reshape correctly.

## D.2 Hyperparameter Failures

| LR | D acc |
|---|---|
| 5e-5 | {{D_LR5e5}} |
| 1e-4 | {{D_LR1e4}} |
| 2e-4 | {{D_ACC}} |
| 5e-4 | {{D_LR5e4}} |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | {{SEED42_D}} | {{SEED42_R}} | {{SEED42_G}} |
| 123 | {{SEED123_D}} | {{SEED123_R}} | {{SEED123_G}} |
| 456 | {{SEED456_D}} | {{SEED456_R}} | {{SEED456_G}} |
| Mean | {{MEAN_D}} | {{MEAN_R}} | {{MEAN_G}} |
| Std | {{STD_D}} | {{STD_R}} | {{STD_G}} |

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
```

---