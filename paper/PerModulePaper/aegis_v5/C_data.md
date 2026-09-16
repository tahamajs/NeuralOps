# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Change | {{D_PARAM_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| SVD energy | {{ENERGY}} |
| Max rank | {{MAX_RANK}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |

## C.4 Rank Ablation

| max_rank | DistilBERT | Params |
|---|---|---|
| 32 | {{D_R32}} | {{D_R32_P}}M |
| 64 | {{D_R64}} | {{D_R64_P}}M |
| 128 | {{D_R128}} | {{D_R128_P}}M |
| 256 | {{D_R256}} | {{D_R256_P}}M |
| 384 | {{D_R384}} | {{D_R384_P}}M |

## C.5 Head Split Ablation

| Split | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| All local | {{D_ALL_LOCAL}} | {{R_ALL_LOCAL}} | {{G_ALL_LOCAL}} |
| 8/4 | {{D_SPLIT84}} | {{R_SPLIT84}} | {{G_SPLIT84}} |
| 6/6 | {{D_SPLIT66}} | {{R_SPLIT66}} | {{G_SPLIT66}} |
| All global | {{D_ALL_GLOBAL}} | {{R_ALL_GLOBAL}} | {{G_ALL_GLOBAL}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
...
50    {{D_S50}}
```

## C.7 Reproduction Commands

```bash
python run_all_4_aegis_v5.py --svd_energy 0.95 --max_rank 128
```

---