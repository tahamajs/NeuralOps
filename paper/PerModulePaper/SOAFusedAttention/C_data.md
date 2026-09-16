# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | {{D_TOTAL_PARAMS}} |
| Attention params before | 2,359,296 |
| Attention params after | {{D_ATTN_PARAMS}} |
| Attention reduction | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | {{R_TOTAL_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | {{G_TOTAL_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Rank Ablation

| $k$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{R_R128}} | {{G_R128}} |
| 192 | {{D_R192}} | {{R_R192}} | {{G_R192}} |
| 384 | {{D_R384}} | {{R_R384}} | {{G_R384}} |

## C.5 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.6 Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.7 Reconstruction Error

| Layer | Q | K | V |
|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} |

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
python run_all_4_soa_fused.py --models distilbert,roberta,gpt2
```
```

---