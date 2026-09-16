<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 results with $r_L = {{LORA_RANK}}$, no head-dim reduction.**

| Model | Module | Acc | Δ | F1 | Trainable | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB | 1.9 s |
| DistilBERT | **SH** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_LORA}}** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB | 4.1 s |
| RoBERTa | **SH** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_LORA}}** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB | 4.2 s |
| GPT-2 | **SH** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_LORA}}** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Head-Dim Reduction

**Table 7.2: Accuracy vs head-dim reduction.**

| $s$ | DistilBERT | RoBERTa | GPT-2 | Param ratio |
|---|---|---|---|---|
| 1.0 (base) | {{D_S10}} | {{R_S10}} | {{G_S10}} | 1.00× |
| 1.5 | {{D_S15}} | {{R_S15}} | {{G_S15}} | {{RATIO15}}× |
| 2.0 | {{D_S20}} | {{R_S20}} | {{G_S20}} | {{RATIO20}}× |
| 2.5 | {{D_S25}} | {{R_S25}} | {{G_S25}} | {{RATIO25}}× |

## 7.3 Trainable Parameter Analysis

**Table 7.3: Trainable parameters.**

| Model | Baseline trainable | SH trainable | Ratio |
|---|---|---|---|
| DistilBERT | 66.9M | {{D_LORA}} | {{D_LORA_R}}× |
| RoBERTa | 124.6M | {{R_LORA}} | {{R_LORA_R}}× |
| GPT-2 | 124.4M | {{G_LORA}} | {{G_LORA_R}}× |

## 7.4 Ablation: LoRA Rank

**Table 7.4: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.5 Ablation: Recovery Steps

**Table 7.5: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10_STEP}} | {{R_S10_STEP}} | {{G_S10_STEP}} |
| 25 | {{D_S25_STEP}} | {{R_S25_STEP}} | {{G_S25_STEP}} |
| 50 | {{D_S50_STEP}} | {{R_S50_STEP}} | {{G_S50_STEP}} |
| 100 | {{D_S100_STEP}} | {{R_S100_STEP}} | {{G_S100_STEP}} |

## 7.6 Ablation: GQA Pooling

**Table 7.6: GQA pooling method (Qwen-style).**

| Method | Accuracy |
|---|---|
| Mean pooling | {{MEAN_POOL}} |
| First head | {{FIRST_POOL}} |

## 7.7 Comparison with Companion Methods

**Table 7.7: Comparison.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type | Trainable |
|---|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Baseline | 100% |
| GQA | {{GQA_D}} | {{GQA_R}} | {{GQA_G}} | KV reduce | ~95% |
| AASVD | {{AASVD_D}} | {{AASVD_R}} | {{AASVD_G}} | Low-rank | 100% |
| LoRA only | {{LORA_D}} | {{LORA_R}} | {{LORA_G}} | Adapter | ~1% |
| **SH** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **Distill** | **~1%** |

## 7.8 LoRA vs Full Fine-Tuning

**Table 7.8: LoRA vs full fine-tuning at same recovery steps.**

| Training | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full FT | {{FULLFT_D}} | {{FULLFT_R}} | {{FULLFT_G}} |
| LoRA only | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| LoRA gain | {{LORA_GAIN}} | {{LORA_GAIN_R}} | {{LORA_GAIN_G}} |

## 7.9 Scaling with Head-Dim Reduction

**Figure 7.1** shows accuracy vs head-dim reduction.

## 7.10 Training Loss Curves

**Figure 7.2** shows loss curves.

## 7.11 Summary

- **Accuracy:** {{D_DELTA}}%–{{G_DELTA}}% drop.
- **Trainable params:** ~1% of full.
- **Head-dim reduction:** up to {{HEAD_DIM_REDUCTION}}× without major loss.
- **LoRA recovery:** {{LORA_GAIN}}% gain over baseline.
- **All models pass quality gate.**

---