# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Params change | +{{D_PARAM_PCT}}% |
| Attention params before | 2,359,296 |
| Attention params after | {{D_ATTN}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Time @ L=128 | {{D_T128}} s |
| Time @ L=2048 | {{D_T2048}} s |
| α final | {{D_ALPHA}} |
| output_scale | {{D_OSCALE}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| α final | {{R_ALPHA}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| α final | {{G_ALPHA}} |

## C.4 Feature Dim Ablation

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} |
| 256 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 512 | {{D_F512}} | {{R_F512}} | {{G_F512}} |

## C.5 Speed vs L

| L | DistilBERT base | DistilBERT LEKAv1 |
|---|---|---|
| 128 | 1.90 s | {{D_T128}} s |
| 256 | 3.50 s | {{D_T256}} s |
| 512 | 12.0 s | {{D_T512}} s |
| 1024 | 45.0 s | {{D_T1024}} s |
| 2048 | 180.0 s | {{D_T2048}} s |

## C.6 Per-Layer α

| Layer | α |
|---|---|
| 0 | {{D_L0_ALPHA}} |
| 1 | {{D_L1_ALPHA}} |
| 2 | {{D_L2_ALPHA}} |
| 3 | {{D_L3_ALPHA}} |
| 4 | {{D_L4_ALPHA}} |
| 5 | {{D_L5_ALPHA}} |

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
python run_all_4_leka_v1.py --models distilbert,roberta,gpt2
```

---
