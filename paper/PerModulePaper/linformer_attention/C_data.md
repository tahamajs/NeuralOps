# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Change | +{{D_PARAM_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| k | {{D_K}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| k | {{R_K}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| k | {{G_K}} |

## C.4 k Ablation

| k | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_K16}} | {{R_K16}} | {{G_K16}} |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} |
| 64 | {{D_K64}} | {{R_K64}} | {{G_K64}} |
| 128 | {{D_K128}} | {{R_K128}} | {{G_K128}} |
| 256 | {{D_K256}} | {{R_K256}} | {{G_K256}} |

## C.5 Init Ablation

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Diagonal + noise | {{D_DIAG}} | {{R_DIAG}} | {{G_DIAG}} |
| Gaussian only | {{D_GAUSS}} | {{R_GAUSS}} | {{G_GAUSS}} |
| Zero | {{D_ZERO}} | {{R_ZERO}} | {{G_ZERO}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

## C.7 Reproduction Commands

```bash
python run_all_4_linformer.py --linformer_k 64
```

---