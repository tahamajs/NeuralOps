# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **BAM-UMoE** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **BAM-UMoE** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **BAM-UMoE** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts. BAM-UMoE EXPANDS the model.**

| Model | Baseline | BAM-UMoE | Increase | Ratio |
|---|---|---|---|---|
| DistilBERT | 66.9M | {{D_PARAMS}}M | +{{D_PARAM_DELTA}}M | {{D_PARAM_RATIO}}× |
| RoBERTa | 124.6M | {{R_PARAMS}}M | +{{R_PARAM_DELTA}}M | {{R_PARAM_RATIO}}× |
| GPT-2 | 124.4M | {{G_PARAMS}}M | +{{G_PARAM_DELTA}}M | {{G_PARAM_RATIO}}× |

**Critical.** BAM-UMoE **increases** parameter count by ~{{AVG_PARAM_INC}}%. 
It is **not a compression method**.

## 7.3 MoE Load Balancing

**Table 7.3: Expert utilization (fraction of tokens per expert).**

| Model | Min utilization | Max utilization | Ideal (1/E) |
|---|---|---|---|
| DistilBERT | {{D_MIN_UTIL}} | {{D_MAX_UTIL}} | 0.125 |
| RoBERTa | {{R_MIN_UTIL}} | {{R_MAX_UTIL}} | 0.125 |
| GPT-2 | {{G_MIN_UTIL}} | {{G_MAX_UTIL}} | 0.125 |

Well-balanced: min/max within ±{{IMBALANCE}}% of ideal.

## 7.4 Efficiency

**Table 7.4: Inference.**

| Model | Config | Time | VRAM |
|---|---|---|---|
| DistilBERT | full | 1.90 s | 183 MB |
| DistilBERT | BAM-UMoE | {{D_TIME}} s | {{D_VRAM}} MB |
| GPT-2 | full | 4.20 s | 793 MB |
| GPT-2 | BAM-UMoE | {{G_TIME}} s | {{G_VRAM}} MB |

## 7.5 Ablation: Number of Experts

**Table 7.5.**

| $E$ | D acc | R acc | G acc |
|---|---|---|---|
| 2 | {{D_E2}} | {{R_E2}} | {{G_E2}} |
| 4 | {{D_E4}} | {{R_E4}} | {{G_E4}} |
| **8** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 16 | {{D_E16}} | {{R_E16}} | {{G_E16}} |

## 7.6 Ablation: Top-K

**Table 7.6.**

| $k$ | D acc | Time overhead |
|---|---|---|
| 1 | {{D_K1}} | 1× |
| **2** | **{{D_ACC}}** | **2×** |
| 4 | {{D_K4}} | 4× |

## 7.7 Ablation: Gate Initialization

**Table 7.7.**

| $b_g$ | D acc | Notes |
|---|---|---|
| 0.0 | {{D_BG0}} | Unstable ramp |
| -1.0 | {{D_BG1}} | Faster ramp |
| **-3.0** | **{{D_ACC}}** | **Default** |
| -5.0 | {{D_BG5}} | Very slow ramp |

## 7.8 Summary

- **Accuracy:** {{D_DELTA}}%–{{G_DELTA}}% change.
- **Parameters:** +{{AVG_PARAM_INC}}% (expansion).
- **Load balancing:** well-behaved.
- **Step-0 preservation:** verified (see Section 8.1).
