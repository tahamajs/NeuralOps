# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Change | +{{D_PARAM_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| L_land | {{D_LLAND}} |
| W | {{D_W}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| L_land | {{R_LLAND}} |
| W | {{R_W}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| L_land | {{G_LLAND}} |
| W | {{G_W}} |

## C.4 L_0 Ablation

| L_0 | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_L8}} | {{R_L8}} | {{G_L8}} |
| 16 | {{D_L16}} | {{R_L16}} | {{G_L16}} |
| 32 | {{D_L32}} | {{R_L32}} | {{G_L32}} |
| 64 | {{D_L64}} | {{R_L64}} | {{G_L64}} |
| 128 | {{D_L128}} | {{R_L128}} | {{G_L128}} |

## C.5 W Ablation

| W | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_W8}} | {{R_W8}} | {{G_W8}} |
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |

## C.6 Feature Dim Ablation

| d_feat | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |

## C.7 Component Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | {{D_FULL}} | {{R_FULL}} | {{G_FULL}} |
| − scorer | {{D_NO_SCORER}} | {{R_NO_SCORER}} | {{G_NO_SCORER}} |
| − local | {{D_NO_LOCAL}} | {{R_NO_LOCAL}} | {{G_NO_LOCAL}} |
| − W_corr | {{D_NO_WCORR}} | {{R_NO_WCORR}} | {{G_NO_WCORR}} |
| − multi-scale | {{D_NO_MULTI}} | {{R_NO_MULTI}} | {{G_NO_MULTI}} |

## C.8 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
...
50    {{D_S50}}
```

## C.9 Reproduction Commands

```bash
python run_all_4_leka_v3.py --num_landmarks 64 --local_window_size 16
```
```

---