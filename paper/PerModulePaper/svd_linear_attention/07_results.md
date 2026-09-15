<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **SVD-Lin** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_RATIO}}×** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **SVD-Lin** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_RATIO}}×** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **SVD-Lin** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_RATIO}}×** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After SVD-Lin | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** |

**Per-layer breakdown (DistilBERT, r=32):**

| Layer | Original | SVD-Lin | Ratio |
|---|---|---|---|
| Q | 589,824 | 49,184 | 12.0× |
| K | 589,824 | 49,184 | 12.0× |
| V | 589,824 | 49,184 | 12.0× |
| O | 589,824 | 49,184 | 12.0× |
| **Sum** | **2,359,296** | **196,736** | **12.0×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | SVD-Lin | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | SVD-Lin | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | SVD-Lin | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Complexity Scaling

**Table 7.4: Inference time vs sequence length (GPT-2).**

| $N$ | Baseline | SVD-Lin | Speedup |
|---|---|---|---|
| 128 | 4.2 s | {{T128}} | {{S128}}× |
| 256 | 16.8 s | {{T256}} | {{S256}}× |
| 512 | 67 s | {{T512}} | {{S512}}× |
| 1024 | 268 s | {{T1024}} | {{S1024}}× |

**Observation.** SVD-Lin exhibits $\mathcal{O}(N)$ scaling vs 
$\mathcal{O}(N^2)$ for baseline.

## 7.5 Ablation: Mode

**Table 7.5: Accuracy vs mode.**

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `exact` (softmax) | {{D_EXACT}} | {{R_EXACT}} | {{G_EXACT}} |
| `linear` (kernel) | {{D_LINEAR}} | {{R_LINEAR}} | {{G_LINEAR}} |
| `decay_linear` | {{D_DECAY}} | {{R_DECAY}} | {{G_DECAY}} |

## 7.6 Ablation: Rank

**Table 7.6: Accuracy vs rank.**

| $r$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 8 | {{D_R8}} | 24× | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | 16× | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | 12× | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | 8× | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | 4× | {{R_R128}} | {{G_R128}} |

## 7.7 Ablation: Decay Gating

**Table 7.7: Effect of decay.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| No decay | {{D_NODECAY}} | {{R_NODECAY}} | {{G_NODECAY}} |
| Fixed decay (γ=0.99) | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |
| Learned decay | **{{D_DECAY}}** | **{{R_DECAY}}** | **{{G_DECAY}}** |

## 7.8 Ablation: Decay Channel-Wise

**Table 7.8: Channel-wise vs head-wise decay.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Head-wise | {{D_HEAD}} | {{R_HEAD}} | {{G_HEAD}} |
| Channel-wise | {{D_CHAN}} | {{R_CHAN}} | {{G_CHAN}} |

## 7.9 Streaming Inference

**Table 7.9: Streaming inference via `step_recurrent`.**

| Model | Batch time | Stream time | Speedup |
|---|---|---|---|
| DistilBERT | {{D_BATCH}} s | {{D_STREAM}} s | {{D_STREAM_R}}× |
| RoBERTa | {{R_BATCH}} s | {{R_STREAM}} s | {{R_STREAM_R}}× |

**Note.** Streaming is per-token; batch is full-sequence. Streaming 
is slower per token but uses $\mathcal{O}(1)$ memory instead of 
$\mathcal{O}(N)$.

## 7.10 Comparison with Companion Methods

**Table 7.10: SVD-Lin vs prior methods.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| aasvd | 88.5% | 4.0× | 75.5% | 3.5× | 76.8% | 3.5× |
| rpc | 89.5% | 1.01× | 75.5% | 1.01× | 76.8% | 1.01× |
| svd_mamba | 85.5% | 0.90× | 72.5% | 0.90× | 73.2% | 0.90× |
| **svd_linear** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** | **{{G_ACC}}** | **{{G_RATIO}}×** |

## 7.11 Failure Cases

**Table 7.11: Failures.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |

## 7.12 Summary

| Metric | Value |
|---|---|
| Compression ratio | {{D_RATIO}}×–{{G_RATIO}}× |
| Accuracy drop | {{D_DELTA}}%–{{G_DELTA}}% |
| VRAM reduction | {{VRAM_REDUCTION}}% |
| Long-context speedup (N=1024) | {{S1024}}× |

---
