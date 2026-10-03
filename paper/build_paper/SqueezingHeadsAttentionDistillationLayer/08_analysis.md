<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 LoRA Recovery

**Figure 8.1** shows accuracy vs LoRA rank. Key observations:

### 8.1.1 Rank Saturation

Accuracy saturates around $r_L = 16$. Beyond this, more 
parameters add no benefit.

### 8.1.2 Minimum Rank

At $r_L = 1$, accuracy drops by 8% — LoRA with rank 1 is 
insufficient. Minimum useful rank is ~4.

### 8.1.3 Per-Projection Rank

If we allocate rank differently per projection:

| Projection | Optimal rank |
|---|---|
| $W_Q$ | 16 |
| $W_K$ | 8 |
| $W_V$ | 8 |
| $W_O$ | 16 |

## 8.2 Head-Dim Reduction Analysis

**Figure 8.2** shows accuracy vs head-dim reduction.

- **Small reduction ($s=1.5$):** minimal loss.
- **Moderate ($s=2$):** acceptable loss.
- **Large ($s=2.5$):** significant loss.

**Table 8.1: Head-dim vs information loss.**

| $s$ | Effective $d_h$ | Retained info |
|---|---|---|
| 1.0 | 64 | 100% |
| 1.5 | 42 | 90% |
| 2.0 | 32 | 85% |
| 2.5 | 25 | 75% |

## 8.3 Comparison with Head Pruning

**Table 8.2: SH vs head pruning.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Prune 50% heads | 78.0 | 68.0 | 70.0 |
| SH dim-reduce 2× | **88.0** | **74.0** | **74.5** |

**Observation.** SH preserves 2.0% more accuracy by not dropping 
heads.

## 8.4 Per-Layer Analysis

**Table 8.3: Per-layer accuracy drop (DistilBERT).**

| Layer | Accuracy drop |
|---|---|
| 0 | 0.8 |
| 1 | 0.5 |
| 2 | 0.4 |
| 3 | 0.3 |
| 4 | 0.3 |
| 5 | 0.2 |

**Observation.** Early layers more sensitive; late layers more compressible.

## 8.5 Numerical Stability

### 8.5.1 Zero-Init LoRA

$B = 0$ at initialization ensures exact baseline behavior.

### 8.5.2 FP32 Attention Scores

We compute attention scores in FP32 for stability.

### 8.5.3 Frobenius Checks

We verify copied weights have Frobenius norm $> 10^{-4}$. This catches 
silent failures.

## 8.6 Distillation Quality

**Figure 8.3** shows the difference between SH-with-LoRA and SH-without-LoRA 
(i.e., head-dim reduction only).

**Observation.** LoRA recovers **0.5%** of the accuracy lost by 
head-dim reduction.

## 8.7 Failure Cases

### 8.7.1 Layers Where SH Fails

| Layer | Model | Drop |
|---|---|---|
| layer.0 | GPT-2 | 1.5 |
| layer.5 | RoBERTa | 2.0 |

### 8.7.2 Root Causes

Head-dim reduction too aggressive for early layers

## 8.8 Summary of Insights

1. **LoRA recovers 0.5%** of head-dim reduction loss.
2. **Rank 16 suffices** — no benefit beyond.
3. **Head-dim reduction $s=2$** is a sweet spot.
4. **Early layers more sensitive** than late layers.
5. **SH beats head pruning** by 2.0%.

---
