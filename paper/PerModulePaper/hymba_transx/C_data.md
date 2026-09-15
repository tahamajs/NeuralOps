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

## C.4 Ablation: Differential

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Standard | {{D_MHSA}} | {{R_MHSA}} | {{G_MHSA}} |
| Diff λ=0.2 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| Diff λ=0.8 | {{D_L08}} | {{R_L08}} | {{G_L08}} |

## C.5 Ablation: Fusion

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| α=1 (attn only) | {{D_ATTN_ONLY}} | {{R_ATTN_ONLY}} | {{G_ATTN_ONLY}} |
| α=0 (ssm only) | {{D_SSM_ONLY}} | {{R_SSM_ONLY}} | {{G_SSM_ONLY}} |
| α=0.5 (fixed) | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |
| α learned | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |

## C.6 LoRA Rank

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## C.7 Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.8 α Per Layer (DistilBERT)

| Layer | α |
|---|---|
| 0 | {{D_L0_ALPHA}} |
| 1 | {{D_L1_ALPHA}} |
| 2 | {{D_L2_ALPHA}} |
| 3 | {{D_L3_ALPHA}} |
| 4 | {{D_L4_ALPHA}} |
| 5 | {{D_L5_ALPHA}} |

## C.9 Training Loss

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

## C.10 Reproduction Commands

```bash
python run_all_4_hymba_transx.py --models distilbert,roberta,gpt2
```

---
