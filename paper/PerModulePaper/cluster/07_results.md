# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | VRAM | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 183 MB | 1.90 s |
| DistilBERT | **ClusterFormer** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_VRAM}} | {{D_TIME}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 712 MB | 4.10 s |
| RoBERTa | **ClusterFormer** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_VRAM}} | {{R_TIME}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 793 MB | 4.20 s |
| GPT-2 | **ClusterFormer** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_VRAM}} | {{G_TIME}} |

## 7.2 Parameter Analysis

| Model | Baseline | ClusterFormer | Δ | Ratio |
|---|---|---|---|---|
| DistilBERT | 66.9M | {{D_PARAMS}}M | +{{D_PARAM_DELTA}}M | {{D_PARAM_RATIO}}× |
| RoBERTa | 124.6M | {{R_PARAMS}}M | +{{R_PARAM_DELTA}}M | {{R_PARAM_RATIO}}× |
| GPT-2 | 124.4M | {{G_PARAMS}}M | +{{G_PARAM_DELTA}}M | {{G_PARAM_RATIO}}× |

**ClusterFormer is not a compression method.** Parameter count increases by 
~{{PARAM_CHANGE}}% (LoRA overhead only).

## 7.3 Attention FLOPs

**Table 7.3: Theoretical attention FLOPs (per layer, per head).**

| Config | FLOPs formula | $L=128$ | $L=512$ |
|---|---|---|---|
| Baseline | $2L^2 d_h$ | 2.1M | 33.6M |
| ClusterFormer | $2KLd_h + 2LWd_h$ | **0.79M** | **7.1M** |
| **Speedup** | — | **2.7×** | **4.7×** |

## 7.4 Ablation: Number of Clusters

| $K$ | D acc | R acc | G acc | FLOPs ratio |
|---|---|---|---|---|
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} | 6× |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} | 4× |
| **16** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **2.7×** |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} | 1.9× |
| 64 | {{D_K64}} | {{R_K64}} | {{G_K64}} | 1.4× |

## 7.5 Ablation: Window Size

| $W$ | D acc | Notes |
|---|---|---|
| 0 | {{D_W0}} | Global only |
| 8 | {{D_W8}} | Very local |
| **32** | **{{D_ACC}}** | **Default** |
| 64 | {{D_W64}} | Broader local |

## 7.6 Ablation: K-Means Iterations

| Iters | D acc | Time overhead |
|---|---|---|
| 1 | {{D_IT1}} | 1× |
| **3** | **{{D_ACC}}** | **3×** |
| 5 | {{D_IT5}} | 5× |
| 10 | {{D_IT10}} | 10× |

## 7.7 Ablation: Temperature

| $\tau$ init | D acc | Learned $\tau$ |
|---|---|---|
| 0.05 | {{D_TAU005}} | {{D_TAU005_L}} |
| **0.1** | **{{D_ACC}}** | **{{D_TAU_L}}** |
| 0.5 | {{D_TAU05}} | {{D_TAU05_L}} |
| 1.0 | {{D_TAU1}} | {{D_TAU1_L}} |

## 7.8 Ablation: Blend Initialization

| $\alpha$ init | D acc | Learned $\alpha_\text{blend}$ |
|---|---|---|
| 0.0 (0.5 blend) | {{D_A0}} | {{D_A0_L}} |
| 0.5 (0.62 blend) | {{D_A05}} | {{D_A05_L}} |
| 2.0 (0.88 blend) | {{D_A2}} | {{D_A2_L}} |

## 7.9 Comparison with Companion Methods

| Method | D acc | Params Δ | Sparse? |
|---|---|---|---|
| full (baseline) | 90.71% | 0 | No |
| `nha` | 90.25% | +0.89% | No |
| `adasplash` | 90.25% | +0.89% | Yes (α-entmax) |
| **ClusterFormer** | **{{D_ACC}}** | **+{{D_PARAM_DELTA_PCT}}%** | **Cluster** |

## 7.10 Attention Quality

**Table 7.10: Cluster assignment quality.**

| Model | Mean cluster entropy | Effective clusters used |
|---|---|---|
| DistilBERT | {{D_ENTROPY}} | {{D_EFF_K}} / {{NUM_CLUSTERS}} |
| RoBERTa | {{R_ENTROPY}} | {{R_EFF_K}} / {{NUM_CLUSTERS}} |
| GPT-2 | {{G_ENTROPY}} | {{G_EFF_K}} / {{NUM_CLUSTERS}} |

High entropy and usage of many clusters indicate good diversity.

---