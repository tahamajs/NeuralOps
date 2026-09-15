# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | {{D_PARAMS}} |
| Attention params before | 2,359,296 |
| Attention params after | {{D_ATTN}} |
| Attention ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Mean spectral gate $g$ | {{D_G_MEAN}} |
| Mean head gate $h$ | {{D_H_MEAN}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | {{R_PARAMS}} |
| Attention params | {{R_ATTN}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Mean $g$ | {{R_G_MEAN}} |
| Mean $h$ | {{R_H_MEAN}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | {{G_PARAMS}} |
| Attention params | {{G_ATTN}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Mean $g$ | {{G_G_MEAN}} |
| Mean $h$ | {{G_H_MEAN}} |

## C.4 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_R32_RATIO}}× | {{R_R32}} | {{R_R32_RATIO}}× |
| 64 | {{D_R64}} | {{D_R64_RATIO}}× | {{R_R64}} | {{R_R64_RATIO}}× |
| 128 | {{D_ACC}} | {{D_RATIO}}× | {{R_ACC}} | {{R_RATIO}}× |
| 192 | {{D_R192}} | {{D_R192_RATIO}}× | {{R_R192}} | {{R_R192_RATIO}}× |
| 256 | {{D_R256}} | {{D_R256_RATIO}}× | {{R_R256}} | {{R_R256_RATIO}}× |

## C.5 Gate Ablations

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full VSpec++ | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| No router | {{D_NOROUTE}} | {{R_NOROUTE}} | {{G_NOROUTE}} |
| No spectral gate | {{D_NOGATE}} | {{R_NOGATE}} | {{G_NOGATE}} |
| No head gate | {{D_NOHEAD}} | {{R_NOHEAD}} | {{G_NOHEAD}} |
| Deterministic | {{D_DET}} | {{R_DET}} | {{G_DET}} |

## C.6 Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.7 Per-Layer Gates (DistilBERT)

| Layer | $g_q$ | $g_k$ | $g_v$ | $h$ |
|---|---|---|---|---|
| 0 | {{D_L0_GQ}} | {{D_L0_GK}} | {{D_L0_GV}} | {{D_L0_H}} |
| 1 | {{D_L1_GQ}} | {{D_L1_GK}} | {{D_L1_GV}} | {{D_L1_H}} |
| 2 | {{D_L2_GQ}} | {{D_L2_GK}} | {{D_L2_GV}} | {{D_L2_H}} |
| 3 | {{D_L3_GQ}} | {{D_L3_GK}} | {{D_L3_GV}} | {{D_L3_H}} |
| 4 | {{D_L4_GQ}} | {{D_L4_GK}} | {{D_L4_GV}} | {{D_L4_H}} |
| 5 | {{D_L5_GQ}} | {{D_L5_GK}} | {{D_L5_GV}} | {{D_L5_H}} |

## C.8 Training Loss

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

## C.9 Reproduction Commands

```bash
python run_all_4_vspec.py --models distilbert,roberta,gpt2
```

---
