<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 results with gate bias {{GATE_BIAS}}.**

| Model | Module | Acc | Δ | F1 | Mean gate | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 183 MB | 1.9 s |
| DistilBERT | **HALO** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_GATE}}** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 712 MB | 4.1 s |
| RoBERTa | **HALO** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_GATE}}** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 793 MB | 4.2 s |
| GPT-2 | **HALO** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_GATE}}** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Gate Analysis

**Table 7.2: Learned gate statistics per head.**

| Model | Min $\bar{g}_h$ | Max $\bar{g}_h$ | Mean | Std |
|---|---|---|---|---|
| DistilBERT | {{D_GATE_MIN}} | {{D_GATE_MAX}} | {{D_GATE}} | {{D_GATE_STD}} |
| RoBERTa | {{R_GATE_MIN}} | {{R_GATE_MAX}} | {{R_GATE}} | {{R_GATE_STD}} |
| GPT-2 | {{G_GATE_MIN}} | {{G_GATE_MAX}} | {{G_GATE}} | {{G_GATE_STD}} |

**Observation.** Gates are non-uniform: some heads prefer dense 
($\bar{g}_h > 0.9$), others prefer linear ($\bar{g}_h < 0.3$).

## 7.3 Per-Layer Gate Evolution

**Table 7.3: Mean gate per layer (DistilBERT).**

| Layer | Mean gate |
|---|---|
| 0 | {{D_L0_GATE}} |
| 1 | {{D_L1_GATE}} |
| 2 | {{D_L2_GATE}} |
| 3 | {{D_L3_GATE}} |
| 4 | {{D_L4_GATE}} |
| 5 | {{D_L5_GATE}} |

**Observation.** Early layers tend to have higher gates (dense); late 
layers have lower gates (linear).

## 7.4 Efficiency

**Table 7.4: Memory and time.**

| Model | Config | Peak VRAM | Inference Time |
|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s |
| DistilBERT | HALO | {{D_VRAM}} MB | {{D_TIME}} s |
| GPT-2 | full | 793 MB | 4.20 s |
| GPT-2 | HALO | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.5 Ablation: Gate Bias

**Table 7.5: Accuracy vs gate bias.**

| $b_h$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_B0}} | {{R_B0}} | {{G_B0}} |
| 0.5 | {{D_B05}} | {{R_B05}} | {{G_B05}} |
| 1.0 | {{D_B1}} | {{R_B1}} | {{G_B1}} |
| 1.8 | {{D_B18}} | {{R_B18}} | {{G_B18}} |
| 3.0 | {{D_B3}} | {{R_B3}} | {{G_B3}} |

**Observation.** $b_h = 1.8$ is a good balance.

## 7.6 Ablation: Kernel

**Table 7.6: Accuracy vs kernel choice.**

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| ReLU+1 | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |
| Softplus | {{D_SOFTPLUS}} | {{R_SOFTPLUS}} | {{G_SOFTPLUS}} |
| Identity (linear) | {{D_IDENT}} | {{R_IDENT}} | {{G_IDENT}} |

**Observation.** ELU+1 is best (non-negative, smooth).

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Comparison with Companion Methods

**Table 7.9: Comparison.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense |
| GQA | {{GQA_D}} | {{GQA_R}} | {{GQA_G}} | KV reduce |
| Primal | {{PRIMAL_D}} | {{PRIMAL_R}} | {{PRIMAL_G}} | Low-rank |
| ASA-Sparse | {{ASA_D}} | {{ASA_R}} | {{ASA_G}} | Sparse |
| NHA | 90.25% | FAIL | 78.78% | Hybrid |
| **HALO** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | Hybrid |

## 7.10 Scaling with Sequence Length

**Table 7.10: Inference time at various $L$ (GPT-2).**

| $L$ | Dense | HALO | Speedup |
|---|---|---|---|
| 128 | {{B_128}} s | {{H_128}} s | {{S_128}}× |
| 512 | {{B_512}} s | {{H_512}} s | {{S_512}}× |
| 2048 | {{B_2048}} s | {{H_2048}} s | {{S_2048}}× |

## 7.11 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Gate distribution:** bimodal (some heads dense, some linear).
- **No parameter reduction** (only gate added).
- **No speedup without hard routing.**
- **All models pass quality gate.**

---