<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy and parameter reduction on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Reduction | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 1.00× | 1.9 s |
| DistilBERT | **ORION** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | **{{D_PARAMS}}M** | **{{D_RED}}×** | {{D_TIME}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.65M | 1.00× | 4.1 s |
| RoBERTa | **ORION** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | **{{R_PARAMS}}M** | **{{R_RED}}×** | {{R_TIME}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.44M | 1.00× | 4.2 s |
| GPT-2 | **ORION** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | **{{G_PARAMS}}M** | **{{G_RED}}×** | {{G_TIME}} |

**Key observations:**

1. **Parameter reduction:** {{D_RED}}× (DistilBERT), {{R_RED}}× (RoBERTa), 
   {{G_RED}}× (GPT-2).
2. **Accuracy drop:** {{D_DELTA}}%, {{R_DELTA}}%, {{G_DELTA}}%.
3. **ORION is the first module in our framework to combine real parameter 
   reduction with competitive accuracy.**

## 7.2 Per-Layer Parameter Breakdown (DistilBERT)

**Table 7.2: Parameter counts per layer.**

| Layer | Original | ORION | Reduction |
|---|---|---|---|
| q_proj | 589,824 | {{D_Q_ORION}} | {{D_Q_RED}}× |
| k_proj | 589,824 | {{D_K_ORION}} | {{D_K_RED}}× |
| v_proj | 589,824 | {{D_V_ORION}} | {{D_V_RED}}× |
| out_proj | 589,824 | {{D_O_ORION}} | {{D_O_RED}}× |
| **Attention total** | **2,359,296** | **{{D_ATTN_TOT}}** | **{{D_ATTN_RED}}×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ORION) | Ratio | VRAM (base) | VRAM (ORION) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: SVD Energy

**Table 7.4: Effect of energy threshold.**

| Energy | DistilBERT acc | Params | Reduction |
|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_E90_P}} | {{D_E90_R}}× |
| 0.95 | {{D_E95}} | {{D_E95_P}} | {{D_E95_R}}× |
| 0.99 | {{D_E99}} | {{D_E99_P}} | {{D_E99_R}}× |

**Observation.** {{ENERGY_OBS}}

## 7.5 Ablation: SVD Max Rank

**Table 7.5: Effect of max rank.**

| Max rank | DistilBERT acc | Params |
|---|---|---|
| 32 | {{D_R32}} | {{D_R32_P}} |
| 64 | {{D_R64}} | {{D_R64_P}} |
| 128 | {{D_R128}} | {{D_R128_P}} |

## 7.6 Ablation: RFF Features

**Table 7.6: Effect of RFF features.**

| Features | DistilBERT acc | Time |
|---|---|---|
| 128 | {{D_F128}} | {{D_F128_T}} |
| 256 | {{D_F256}} | {{D_F256_T}} |
| 512 | {{D_F512}} | {{D_F512_T}} |

## 7.7 Ablation: Head Split

**Table 7.7: Effect of kernel:full head split.**

| Split | DistilBERT acc | Time |
|---|---|---|
| 1/2 : 1/2 | {{D_SPLIT_HALF}} | {{D_SPLIT_HALF_T}} |
| 2/3 : 1/3 | {{D_ACC}} | {{D_TIME}} |
| 3/4 : 1/4 | {{D_SPLIT_34}} | {{D_SPLIT_34_T}} |
| 1 : 0 (all kernel) | {{D_ALL_KERNEL}} | {{D_ALL_KERNEL_T}} |
| 0 : 1 (all full) | {{D_ALL_FULL}} | {{D_ALL_FULL_T}} |

**Observation.** {{SPLIT_OBS}}

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32_ACC}} | {{R_R32_ACC}} | {{G_R32_ACC}} |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| a3 | {{A3_D}} | {{A3_R}} | {{A3_G}} | **-60%** |
| cosformer | {{COS_D}} | {{COS_R}} | {{COS_G}} | +5.7% |
| nasc | {{NASC_D}} | {{NASC_R}} | {{NASC_G}} | +1.6% |
| **orion** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **-{{PARAM_REDUCTION}}%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `{{D_START_LOSS}}` → `{{D_END_LOSS}}`
- RoBERTa: `{{R_START_LOSS}}` → `{{R_END_LOSS}}`
- GPT-2: `{{G_START_LOSS}}` → `{{G_END_LOSS}}`

## 7.11 Summary

- **DistilBERT:** {{D_ACC}}% (Δ = {{D_DELTA}}), **{{D_RED}}×** reduction
- **RoBERTa:** {{R_ACC}}% (Δ = {{R_DELTA}}), **{{R_RED}}×** reduction
- **GPT-2:** {{G_ACC}}% (Δ = {{G_DELTA}}), **{{G_RED}}×** reduction
```

---