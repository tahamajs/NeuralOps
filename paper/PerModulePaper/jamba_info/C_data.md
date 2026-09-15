# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Params change | +{{D_PARAM_PCT}}% |
| Trainable | {{D_TRAINABLE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| α final | {{D_ALPHA}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| α final | {{R_ALPHA}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| α final | {{G_ALPHA}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Bottleneck Dim Ablation

| $b$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | {{D_B64}} | {{R_B64}} | {{G_B64}} |
| 128 | {{D_B128}} | {{R_B128}} | {{G_B128}} |
| 256 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 512 | {{D_B512}} | {{R_B512}} | {{G_B512}} |

## C.5 No-Bottleneck Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With BN | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| No BN | {{D_NOBN}} | {{R_NOBN}} | {{G_NOBN}} |

## C.6 α Init Ablation

| $w_0$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| -4 | {{D_WM4}} | {{R_WM4}} | {{G_WM4}} |
| -2 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 0 | {{D_W0}} | {{R_W0}} | {{G_W0}} |
| +2 | {{D_WP2}} | {{R_WP2}} | {{G_WP2}} |

## C.7 Training Loss

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

## C.8 Reproduction Commands

```bash
python run_all_4_jamba_info.py --models distilbert,roberta,gpt2
```

---
