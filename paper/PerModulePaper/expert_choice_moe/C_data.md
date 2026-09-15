# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | {{D_PARAMS}} |
| Trainable | {{D_TRAINABLE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Expert Count Ablation

| $E$ | DistilBERT | RoBERTa | GPT-2 | Params/layer |
|---|---|---|---|---|
| 4 | {{D_E4}} | {{R_E4}} | {{G_E4}} | 9.4M |
| 8 | {{D_E8}} | {{R_E8}} | {{G_E8}} | 18.9M |
| 16 | {{D_E16}} | {{R_E16}} | {{G_E16}} | 37.8M |

## C.5 Capacity Factor Ablation

| CF | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_CF05}} | {{R_CF05}} | {{G_CF05}} |
| 1.0 | {{D_CF10}} | {{R_CF10}} | {{G_CF10}} |
| 1.5 | {{D_CF15}} | {{R_CF15}} | {{G_CF15}} |
| 2.0 | {{D_CF20}} | {{R_CF20}} | {{G_CF20}} |

## C.6 MoE Gate Initialization Ablation

| $b_g$ | σ($b_g$) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -8.0 | 0.0003 | {{D_G_M8}} | {{R_G_M8}} | {{G_G_M8}} |
| -4.0 | 0.018 | {{D_G_M4}} | {{R_G_M4}} | {{G_G_M4}} |
| -2.0 | 0.12 | {{D_G_M2}} | {{R_G_M2}} | {{G_G_M2}} |
| 0.0 | 0.50 | {{D_G_0}} | {{R_G_0}} | {{G_G_0}} |

## C.7 Expert Utilization

| Expert | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 12.5% | 12.5% | 12.5% |
| 1 | 12.5% | 12.5% | 12.5% |
| ... | ... | ... | ... |
| 7 | 12.5% | 12.5% | 12.5% |

**Perfect balance.**

## C.8 Per-Token Capacity Distribution

| Selected by | Fraction |
|---|---|
| 0 experts | {{PCT_0}}% |
| 1 expert | {{PCT_1}}% |
| 2 experts | {{PCT_2}}% |
| 3+ experts | {{PCT_3}}% |

## C.9 Gate Evolution

| Step | γ |
|---|---|
| 0 | 0.018 |
| 10 | {{GAMMA_STEP10}} |
| 25 | {{GAMMA_STEP25}} |
| 50 | {{GAMMA_STEP50}} |

## C.10 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
20    {{D_S20}}
30    {{D_S30}}
40    {{D_S40}}
50    {{D_S50}}
```

## C.11 Reproduction Commands

```bash
python run_all_4_expert_choice_moe.py --models distilbert,roberta,gpt2
```

---
