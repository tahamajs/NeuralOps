# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params | Path |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | — |
| DistilBERT | **Galerkin** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | Galerkin |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | — |
| RoBERTa | **Galerkin** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | Galerkin |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | — |
| GPT-2 | **Galerkin** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | Fallback |

## 7.2 Parameter Analysis

| Model | Baseline | Galerkin | Δ |
|---|---|---|---|
| DistilBERT | 66.9M | {{D_PARAMS}}M | +{{D_PARAM_DELTA}}M |
| RoBERTa | 124.6M | {{R_PARAMS}}M | +{{R_PARAM_DELTA}}M |
| GPT-2 | 124.4M | {{G_PARAMS}}M | +{{G_PARAM_DELTA}}M |

**Galerkin Attention increases parameter count by ~{{PARAM_CHANGE}}%.**

## 7.3 Path Analysis

**Table 7.3: Path taken by model.**

| Model | Path | Attention complexity |
|---|---|---|
| DistilBERT | Galerkin | $O(N r d)$ |
| RoBERTa | Galerkin | $O(N r d)$ |
| GPT-2 | Fallback | $O(N^2 d)$ |

## 7.4 Attention FLOPs

**Table 7.4: Theoretical attention FLOPs per layer.**

| $N$ | GPT-2 (fallback) | Encoder (Galerkin, $r=64$) |
|---|---|---|
| 64 | 16K | 16K |
| 128 | 65K | 65K |
| 256 | 262K | 131K |
| 512 | 1.05M | 262K |
| 1024 | 4.2M | 524K |

## 7.5 Ablation: Rank

**Table 7.5 (encoder only).**

| $r$ | D acc | R acc | FLOPs vs baseline |
|---|---|---|---|
| 16 | {{D_R16}} | {{R_R16}} | 8× |
| 32 | {{D_R32}} | {{R_R32}} | 4× |
| **64** | **{{D_ACC}}** | **{{R_ACC}}** | **2×** |
| 128 | {{D_R128}} | {{R_R128}} | 1× |

## 7.6 Ablation: LoRA Rank

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} |
| **16** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} |

## 7.7 Comparison with Companion Methods

| Method | D acc | R acc | G acc |
|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% |
| `nha` | 90.25% | — | 78.78% |
| `adasplash` | 90.25% | — | 78.78% |
| **Galerkin** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

## 7.8 Summary

- **DistilBERT:** {{D_ACC}}% (encoder Galerkin)
- **RoBERTa:** {{R_ACC}}% (encoder Galerkin)
- **GPT-2:** {{G_ACC}}% (decoder fallback — same as baseline)
