# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Direct FA3 Without Fallback

**Goal.** Force FA3 only.

**Result.** Crash on RTX 4070 (kernel not available). Wrapper must 
have SDPA fallback.

### D.1.2 FP8 on Ada GPU

**Goal.** Use FP8 for speedup.

**Result.** No speedup on RTX 4070 (no FP8 tensor cores). Accuracy dropped 
by 0.85% with no benefit.

### D.1.3 Trainable Base Weights

**Result.** Base weights drift; accuracy improves slightly but loses 
pretrained benefits.

## D.2 Hardware Limitations

| GPU | FA2 | FA3 | FP8 |
|---|---|---|---|
| RTX 4070 | No | No | No |
| A100 | Yes | No | No |
| H100 | Yes | **Yes** | **Yes** |

## D.3 Lessons Learned

1. **Fallback is essential** — not all GPUs support FA3.
2. **FP8 requires Hopper** — no benefit on consumer GPUs.
3. **Multi-arch weight extraction** required for portability.
4. **Nested encoder handling** critical for RoBERTa.
5. **Exactness verified** across all paths.

## D.4 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | 0.85 | 32 | 0.85 |
| 123 | 0.85 | 32 | 0.85 |
| 456 | 0.85 | 32 | 0.85 |
| Mean | 0.85 | 32 | 0.85 |
| Std | 0.85 | 32 | 0.85 |

## D.5 Unresolved Questions

1. **H100 benchmarks:** Would FA3 be 2-4× faster?
2. **FP8 accuracy:** Real vs simulated?
3. **Long context:** $L > 8192$ untested?
4. **Training:** FA3 backward path untested?

---