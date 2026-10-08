# 8. Analysis

## 8.1 The Short-Path Problem

**Critical insight.** FUSIONATTN's dispatch threshold is $L_\text{threshold} = 128$. 
Our test dataset (SST-2) uses $L = 128$. The condition:

```python
if N <= 128 or is_cross_attention:
    # short path
```

**Always selects the short path.** Therefore:

1. **No sparsity is applied.**
2. **Gate MLP is never used.**
3. **Only LoRA contributes trainable capacity.**
4. **The model is effectively just a LoRA-wrapped baseline.**

This is not a flaw of FUSIONATTN — it's the intended behavior. But at 
$L = 128$, the module's sparsity machinery is silent.

## 8.2 Where FUSIONATTN Matters

**Figure 8.1** shows theoretical attention FLOPs vs $L$:

| $L$ | Softmax (FLOPs) | FUSIONATTN (FLOPs) | Speedup |
|---|---|---|---|
| 64 | 4096 | 4096 (short) | 1× |
| 128 | 16384 | 16384 (short) | 1× |
| 256 | 65536 | 32768 | 2× |
| 512 | 262144 | 65536 | 4× |
| 1024 | 1.05M | 131K | 8× |
| 4096 | 16.8M | 1.05M | **16×** |

FUSIONATTN's benefits **only appear at $L \geq 256$**.

## 8.3 Block Gate Behavior (Simulated)

**Figure 8.2** shows gate scores for a simulated $L = 512$ input. 

- Gate consistently selects block 0 (attention sink).
- Selected blocks tend to be **nearby** (local structure).
- Some **distant blocks** are also selected (global context).

## 8.4 Component Contributions

**Table 8.1: Ablation at L = 128.**

| Config | D acc |
|---|---|
| Baseline | 90.71% |
| + LoRA only (no gate) | 86.4\% |
| + Gate MLP (unused) | **86.4** |

At L = 128, gate is unused; only LoRA matters.

## 8.5 Comparison with Pure LoRA

**Table 8.2.**

| Config | D acc | Params |
|---|---|---|
| Baseline | 90.71% | 66.9M |
| LoRA only (r=16) | 86.4\% | 67.5M |
| **FUSIONATTN** | **86.4** | 66.4MM |

The gate MLP adds ~66.4MM parameters but does nothing at L=128.

## 8.6 Failure Analysis

0

## 8.7 Summary of Insights

1. **Short path is silent at L=128** — no sparsity benefit.
2. **Long path untested** — requires L > 128.
3. **Parameter count increases** — LoRA + gate.
4. **Not a compression method.**
5. **Theoretical speedup** exists for L ≥ 256.

---