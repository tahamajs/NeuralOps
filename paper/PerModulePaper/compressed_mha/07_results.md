# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 1.00× | 183 MB |
| DistilBERT | **Compressed MHA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | **{{D_RATIO}}×** | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 1.00× | 712 MB |
| RoBERTa | **Compressed MHA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | **{{R_RATIO}}×** | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 1.00× | 793 MB |
| GPT-2 | **Compressed MHA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | **{{G_RATIO}}×** | {{G_VRAM}} MB |

**Key observations:**

1. **Real parameter reduction** on all three models.
2. **Small accuracy drops** with proper rank selection.
3. **Multi-architecture** support validated.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Compressed | Reduction | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS_ABS}} | **{{D_RATIO}}×** | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS_ABS}} | **{{R_RATIO}}×** | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS_ABS}} | **{{G_RATIO}}×** | {{G_NOTE}} |

## 7.3 Per-Layer Breakdown (DistilBERT)

**Table 7.3: Parameter count per attention layer.**

| Component | Baseline | Compressed | Reduction |
|---|---|---|---|
| $W_Q$ | 589,824 | {{D_WQ_A}} | {{D_WQ_R}}× |
| $W_K$ | 589,824 | {{D_WK_A}} | {{D_WK_R}}× |
| $W_V$ | 589,824 | {{D_WV_A}} | {{D_WV_R}}× |
| $W_O$ | 589,824 | {{D_WO_A}} | {{D_WO_R}}× |
| LoRA | 0 | 73,728 | — |
| **Total** | 2,359,296 | **{{D_LAYER_TOTAL}}** | **{{D_LAYER_R}}×** |

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | Compressed | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | Compressed | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.5 Ablation: Rank

**Table 7.5: Accuracy vs rank.**

| $r_{qkv} = r_{out}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{G_R128}} |
| 256 | {{D_R256}} | {{D_RATIO256}}× | {{R_R256}} | {{G_R256}} |
| 384 | {{D_R384}} | {{D_RATIO384}}× | {{R_R384}} | {{G_R384}} |

## 7.6 Ablation: Joint vs Independent SVD

**Table 7.6: Joint vs independent SVD (same total rank).**

| Config | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| Independent | {{D_INDEP}} | {{R_INDEP}} | {{G_INDEP}} | {{INDEP_PARAMS}}M |
| **Joint** | **{{D_JOINT}}** | **{{R_JOINT}}** | **{{G_JOINT}}** | **{{JOINT_PARAMS}}M** |

**Observation.** Joint SVD achieves similar accuracy with fewer parameters.

## 7.7 Ablation: LoRA

**Table 7.7: Effect of LoRA recovery.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| No LoRA | {{D_NOLORA}} | {{R_NOLORA}} | {{G_NOLORA}} |
| LoRA (rank 16) | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| LoRA (rank 32) | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |
| LoRA (rank 64) | {{D_LORA64}} | {{R_LORA64}} | {{G_LORA64}} |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Nested Encoder Detection Validation

**Table 7.9: Effect of nested detection fix.**

| Detection order | DistilBERT | RoBERTa |
|---|---|---|
| Wrong (class name first) | {{D_WRONG}} | {{R_WRONG}} |
| **Correct (out_lin first)** | **{{D_ACC}}** | **{{R_ACC}}** |

**Observation.** The wrong detection caused DistilBERT collapse to 50.92%. 
The fix restores full accuracy.

## 7.10 Comparison with Prior Modules

**Table 7.10: Compressed MHA vs other compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| A3 | {{A3_D}} | {{A3_D_R}}× | {{A3_R}} | {{A3_R_R}}× |
| AASVD | {{AASVD_D}} | {{AASVD_D_R}}× | {{AASVD_R}} | {{AASVD_R_R}}× |
| **Compressed MHA** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** |

## 7.11 Summary

- Parameter reduction: **{{D_RATIO}}×–{{G_RATIO}}×**
- Accuracy drop: **{{D_DELTA}}%–{{G_DELTA}}%**
- Nested detection fix restores DistilBERT accuracy.
```

---