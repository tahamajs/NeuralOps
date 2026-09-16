# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **Res-SVD** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_RATIO}}×** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **Res-SVD** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_RATIO}}×** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **Res-SVD** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_RATIO}}×** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After Res-SVD | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** |

**Per-layer breakdown (DistilBERT, $r_b=16$, $r_r=4$):**

| Component | Params |
|---|---|
| Base SVD | 24,592 |
| Residual LoRA | 6,144 |
| Bias | 768 |
| **Total per projection** | **31,504** |
| Baseline per projection | 589,824 |
| **Ratio** | **18.7×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Res-SVD | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Res-SVD | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Res-SVD | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Base Rank

**Table 7.4: Accuracy vs base rank ($r_r = 4$).**

| $r_b$ | DistilBERT | RoBERTa | GPT-2 | Ratio |
|---|---|---|---|---|
| 8 | {{D_B8}} | {{R_B8}} | {{G_B8}} | 25× |
| 16 | {{D_B16}} | {{R_B16}} | {{G_B16}} | 18.7× |
| 32 | {{D_B32}} | {{R_B32}} | {{G_B32}} | 14× |
| 64 | {{D_B64}} | {{R_B64}} | {{G_B64}} | 8× |

## 7.5 Ablation: Residual Rank

**Table 7.5: Accuracy vs residual rank ($r_b = 16$).**

| $r_r$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_R0}} | {{R_R0}} | {{G_R0}} |
| 2 | {{D_R2}} | {{R_R2}} | {{G_R2}} |
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |

**Observation.** $r_r = 4$ is a good balance.

## 7.6 Ablation: Scaling $\alpha$

**Table 7.6: Accuracy vs $\alpha$.**

| $\alpha$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_A05}} | {{R_A05}} | {{G_A05}} |
| 1.0 | {{D_A10}} | {{R_A10}} | {{G_A10}} |
| 2.0 | {{D_A20}} | {{R_A20}} | {{G_A20}} |

## 7.7 Comparison with SVD-Base Only

**Table 7.7: Res-SVD vs pure SVD at equal total rank.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Pure SVD ($r=20$) | {{D_SVD20}} | {{R_SVD20}} | {{G_SVD20}} |
| **Res-SVD ($r_b=16, r_r=4$)** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

**Note.** As proven in §4.7.1, both are mathematically equivalent at 
initialization. Differences arise during training.

## 7.8 Comparison with Companion Methods

**Table 7.8: Res-SVD vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| asvd ($r=32$) | 88.5% | 12× | 75.5% | 11× | 76.8% | 11× |
| aasvd ($r=64$) | 88.5% | 5× | 75.5% | 5× | 76.8% | 5× |
| **res_svd** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** | **{{G_ACC}}** | **{{G_RATIO}}×** |

## 7.9 Failure Cases

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |

---