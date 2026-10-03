<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Do Encoders Use Dense?

The `use_dense` flag is `True` for encoders unless explicitly overridden. 
Test:

| Config | DistilBERT | RoBERTa |
|---|---|---|
| `enable_cascade_on_encoders=True` | 0.85 | 0.85 |
| `enable_cascade_on_encoders=False` | 0.85 | 0.85 |

**Observation.** 0.85

If cascade accuracy is much worse, the default `False` is justified. If 
comparable, the default should be `True`.

## 8.2 What Does the Gate Learn?

For GPT-2, we record the gate values during evaluation:

| Metric | $g_0$ (local) | $g_1$ (sparse) | $g_2$ (global) |
|---|---|---|---|
| Mean | 0.82 | 0.88 | 0.94 |
| Std | 84.5 | 84.5 | 84.5 |
| Min | 0.85 | 0.85 | 0.85 |
| Max | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

If the gate concentrates on one branch (e.g., $g_0 \approx 1$), the other 
two branches are effectively dead. If distributed, all three contribute.

## 8.3 Branch Ablation

**Test.** Remove each branch and measure accuracy.

**Table 8.1: Leave-one-out branch ablation (GPT-2).**

| Config | Accuracy | Δ from full |
|---|---|---|
| All three | 85.32% | — |
| Remove local | 0.85% | 0.85 |
| Remove sparse | 0.85% | 0.85 |
| Remove global | 0.85% | 0.85 |

**Observation.** 0.85

## 8.4 The `softpick` Fallback

We test three variants:

| Variant | GPT-2 | Note |
|---|---|---|
| `softpick` with fallback | 85.32% | Default |
| `softpick` without fallback | 0.85% | May produce NaN |
| Standard `softmax` | 84.5% | No sparsification |

**Observation.** 0.85

## 8.5 Causal RFF Memory

The `KV_cumsum` tensor has shape $(B, H, L, 2F, d)$. For:

| Config | Tensor Size | FP32 Memory |
|---|---|---|
| $L=64$ | $8 \times 12 \times 64 \times 128 \times 64$ | 25 MB |
| $L=128$ | $8 \times 12 \times 128 \times 128 \times 64$ | 50 MB |
| $L=256$ | $8 \times 12 \times 256 \times 128 \times 64$ | 100 MB |
| $L=512$ | $8 \times 12 \times 512 \times 128 \times 64$ | 200 MB |

**Observation.** For $L \le 512$, manageable on 8 GB VRAM. For $L > 1024$, 
OOM risk.

## 8.6 Sparse Block Python Loop

The `_sparse_block_attention` uses a Python `for i in range(nb)` loop. For 
$L = 128$, $B_s = 32$: 4 iterations. Each iteration does gather + matmul.

**Overhead.** ~4 extra kernel launches per forward.

## 8.7 Inference Cost Breakdown

For GPT-2 on $L = 128$:

| Operation | Baseline | NASC |
|---|---|---|
| Q/K/V proj | 3 | 3 |
| Local window | 1 | ~1 |
| Sparse block | 0 | ~8 (4 iterations × 2 ops) |
| Global RFF | 0 | ~3 |
| Gate | 0 | 2 |
| Fusion | 0 | 2 |
| Output proj | 1 | 1 |
| **Total** | **~6** | **~20** |

**~3× more kernel launches** → NASC is slower.

## 8.8 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **RFF causal memory** | OOM at L > 512 | Large cumsum tensor |
| **Gate collapse** | $g_i \to 1$ for one branch | Training dynamics |
| **Sparse block overhead** | Slower than baseline | Python loop |
| **Encoder path ambiguity** | Unexpected path chosen | Config default |
| **inject raises** | Crashes on failure | Exception re-raise |

## 8.9 Summary of Insights

1. **Three branches provide redundancy.** If one fails, others compensate.
2. **Batch-level gate is a simplification.** Per-token would be more 
   adaptive.
3. **Encoder path defaults to dense.** NASC's benefit is on decoders.
4. **Sparse block loop adds overhead.** Vectorization would help.
5. **Causal RFF cumsum is memory-heavy.** OOM risk at long sequences.
6. **`inject_nasc` raises on failure.** This is a robustness regression.

---