<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params Δ % | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **RPC** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **RPC** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **RPC** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After RPC | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_DELTA}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_DELTA}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_DELTA}} | +{{G_PARAM_PCT}}% |

**Note.** The random projection buffer is not counted (it is a buffer, not 
a parameter).

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | RPC | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | RPC | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | RPC | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Sketch Rank

**Table 7.4: Accuracy vs sketch rank.**

| $r$ | DistilBERT | RoBERTa | GPT-2 | Theoretical speedup |
|---|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} | 8× |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} | 4× |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} | 2× |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} | 1× |

**Note.** $r = 64$ is full rank for DistilBERT (head_dim = 64), so no 
sketching occurs.

## 7.5 Ablation: LoRA Rank

**Table 7.5: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_LORA0}} | {{R_LORA0}} | {{G_LORA0}} |
| 8 | {{D_LORA8}} | {{R_LORA8}} | {{G_LORA8}} |
| 16 | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| 32 | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |

## 7.6 Comparison with Companion Methods

**Table 7.6: RPC vs prior methods.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| asvd (r=64) | 88.5% | 75.5% | 76.8% | 0.18× |
| **rpc (r=32)** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **1.01×** |

## 7.7 Actual vs Theoretical Speedup

**Table 7.7: Theoretical vs measured inference time (DistilBERT, L=128).**

| Config | Theoretical | Measured | Gap |
|---|---|---|---|
| r=8 | 8× | {{D_ACTUAL_8}}× | {{D_GAP_8}}× |
| r=16 | 4× | {{D_ACTUAL_16}}× | {{D_GAP_16}}× |
| r=32 | 2× | {{D_ACTUAL_32}}× | {{D_GAP_32}}× |
| r=64 | 1× | 1× | 0 |

**Observation.** The theoretical speedup is not realized in practice 
because:
1. The projection adds overhead.
2. Kernel launch overhead dominates at $L = 128$.
3. Memory bandwidth is the bottleneck, not FLOPs.

## 7.8 Failure Cases

**Table 7.8: Failures.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |

## 7.9 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | {{BEST_DELTA}}% |
| Parameter increase | +{{PARAM_CHANGE}}% |
| Theoretical FLOPs speedup (r=32) | 2× |
| Actual speedup (r=32, L=128) | {{ACTUAL_SPEEDUP}}× |
| Memory reduction | **0%** |
```

---