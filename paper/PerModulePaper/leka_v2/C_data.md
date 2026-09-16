# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Params change | +{{D_PARAM_PCT}}% |
| Attention params | {{D_ATTN}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Landmark ratio | {{D_LR}} |
| Mean local gate | {{D_G_MEAN}} |
| Peak VRAM | {{D_VRAM}} MB |
| Time @ L=128 | {{D_T128}} s |
| Time @ L=1024 | {{D_T1024}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Landmark ratio | {{R_LR}} |
| Mean gate | {{R_G_MEAN}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Landmark ratio | {{G_LR}} |
| Mean gate | {{G_G_MEAN}} |

## C.4 ρ Ablation

| ρ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.25 | {{D_RHO25}} | {{R_RHO25}} | {{G_RHO25}} |
| 0.50 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 0.75 | {{D_RHO75}} | {{R_RHO75}} | {{G_RHO75}} |
| 1.00 | {{D_RHO100}} | {{R_RHO100}} | {{G_RHO100}} |

## C.5 Feature Dim

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |
| 512 | {{D_F512}} | {{R_F512}} | {{G_F512}} |

## C.6 Window Size

| W | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 128 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 256 | {{D_W256}} | {{R_W256}} | {{G_W256}} |

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
python run_all_4_leka_v2.py --models distilbert,roberta,gpt2
```

---