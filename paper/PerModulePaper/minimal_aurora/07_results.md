# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **MinimalAURORA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_RATIO}}×** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **MinimalAURORA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_RATIO}}×** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **MinimalAURORA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_RATIO}}×** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** |

**Per-layer breakdown (DistilBERT, $\varepsilon=0.95$):**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| Rank | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| Params | {{D_L0_Q_P}} | {{D_L0_K_P}} | {{D_L0_V_P}} | {{D_L0_O_P}} |

## 7.3 Rank Distribution

**Table 7.3: Energy-selected rank per layer.**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| 1 | {{D_L1_Q_R}} | {{D_L1_K_R}} | {{D_L1_V_R}} | {{D_L1_O_R}} |
| ... | ... | ... | ... | ... |
| 5 | {{D_L5_Q_R}} | {{D_L5_K_R}} | {{D_L5_V_R}} | {{D_L5_O_R}} |
| **Mean** | **{{D_MEAN_R}}** | | | |

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Ours | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Ours | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Ours | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.5 Ablation: Energy Threshold

**Table 7.5: Accuracy vs $\varepsilon$.**

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_E90_R}}× | {{R_E90}} | {{R_E90_R}}× | {{G_E90}} | {{G_E90_R}}× |
| 0.95 | {{D_E95}} | {{D_E95_R}}× | {{R_E95}} | {{R_E95_R}}× | {{G_E95}} | {{G_E95_R}}× |
| 0.99 | {{D_E99}} | {{D_E99_R}}× | {{R_E99}} | {{R_E99_R}}× | {{G_E99}} | {{G_E99_R}}× |
| 0.995 | {{D_E995}} | {{D_E995_R}}× | {{R_E995}} | {{R_E995_R}}× | {{G_E995}} | {{G_E995_R}}× |

## 7.6 Ablation: Max Rank Cap

**Table 7.6: Accuracy vs rank cap.**

| Cap | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_C16}} | {{R_C16}} | {{G_C16}} |
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} |
| 128 | {{D_C128}} | {{R_C128}} | {{G_C128}} |

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_LORA0}} | {{R_LORA0}} | {{G_LORA0}} |
| 8 | {{D_LORA8}} | {{R_LORA8}} | {{G_LORA8}} |
| 16 | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| 32 | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |

## 7.8 `output_scale` Tracking

**Table 7.8: Final `output_scale` per model.**

| Model | Init | Final | Drift |
|---|---|---|---|
| DistilBERT | 1.00 | {{D_OS}} | {{D_OS_DRIFT}} |
| RoBERTa | 1.00 | {{R_OS}} | {{R_OS_DRIFT}} |
| GPT-2 | 1.00 | {{G_OS}} | {{G_OS_DRIFT}} |

## 7.9 Comparison with Companion Methods

**Table 7.9: Comparison.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| asvd ($r=32$) | 88.5% | 12× | 75.5% | 11× | 76.8% | 11× |
| aasvd ($r=64$) | 88.5% | 5× | 75.5% | 5× | 76.8% | 5× |
| a3 ($r=64$) | 88.5% | 5× | 75.5% | 5× | 76.8% | 5× |
| res_svd (16,4) | 88.2% | 5.9× | 74.8% | 5.5× | 75.5% | 5.5× |
| **minimal_aurora** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** | **{{G_ACC}}** | **{{G_RATIO}}×** |

## 7.10 Failure Cases

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |
```

---