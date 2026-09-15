# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | {{D_TOTAL}} |
| Attention params before | 2,359,296 |
| Attention params after | {{D_ATTN}} |
| Attention ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Avg rank | {{AVG_RANK}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | {{R_TOTAL}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Avg rank | {{R_AVG_RANK}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | {{G_TOTAL}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Avg rank | {{G_AVG_RANK}} |

## C.4 Per-Layer Rank

| Layer | Q r* | K r* | V r* | O r* |
|---|---|---|---|---|
| 0 | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| 1 | {{D_L1_Q_R}} | {{D_L1_K_R}} | {{D_L1_V_R}} | {{D_L1_O_R}} |
| 2 | {{D_L2_Q_R}} | {{D_L2_K_R}} | {{D_L2_V_R}} | {{D_L2_O_R}} |
| 3 | {{D_L3_Q_R}} | {{D_L3_K_R}} | {{D_L3_V_R}} | {{D_L3_O_R}} |
| 4 | {{D_L4_Q_R}} | {{D_L4_K_R}} | {{D_L4_V_R}} | {{D_L4_O_R}} |
| 5 | {{D_L5_Q_R}} | {{D_L5_K_R}} | {{D_L5_V_R}} | {{D_L5_O_R}} |

## C.5 Energy τ Ablation

| τ | DistilBERT | Avg Rank | Ratio |
|---|---|---|---|
| 0.80 | {{D_T80}} | {{D_T80_R}} | {{RATIO_80}}× |
| 0.85 | {{D_T85}} | {{D_T85_R}} | {{RATIO_85}}× |
| 0.92 | {{D_T92}} | {{AVG_RANK}} | {{D_RATIO}}× |
| 0.95 | {{D_T95}} | {{D_T95_R}} | {{RATIO_95}}× |
| 0.99 | {{D_T99}} | {{D_T99_R}} | {{RATIO_99}}× |

## C.6 Residual Rank Ablation

| $r_a$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_RA0}} | {{R_RA0}} | {{G_RA0}} |
| 8 | {{D_RA8}} | {{R_RA8}} | {{G_RA8}} |
| 16 | {{D_RA16}} | {{R_RA16}} | {{G_RA16}} |
| 32 | {{D_RA32}} | {{R_RA32}} | {{G_RA32}} |

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
python run_all_4_adaptive_svd.py --models distilbert,roberta,gpt2
```

---
