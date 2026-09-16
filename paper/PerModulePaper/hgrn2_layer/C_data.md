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
| Peak VRAM | {{D_VRAM}} MB |
| Time (L=128) | {{D_T128}} s |
| Time (L=1024) | {{D_T1024}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Speed vs L

| L | Baseline | Ours |
|---|---|---|
| 128 | 1.90 s | {{D_T128}} s |
| 256 | 3.50 s | {{D_T256}} s |
| 512 | 12.0 s | {{D_T512}} s |
| 1024 | 45.0 s | {{D_T1024}} s |
| 2048 | 180.0 s | {{D_T2048}} s |
| 4096 | 720.0 s | {{D_T4096}} s |

## C.5 Chunk Size Ablation

| Chunk | DistilBERT | Time (L=512) |
|---|---|---|
| 8 | {{D_C8}} | {{D_C8_T}} s |
| 16 | {{D_C16}} | {{D_C16_T}} s |
| 32 | {{D_C32}} | {{D_C32_T}} s |
| 64 | {{D_C64}} | {{D_C64_T}} s |

## C.6 Forget Gate Statistics

| Layer | Mean f | Std f |
|---|---|---|
| 0 | {{D_L0_F_MEAN}} | {{D_L0_F_STD}} |
| 3 | {{D_L3_F_MEAN}} | {{D_L3_F_STD}} |
| 5 | {{D_L5_F_MEAN}} | {{D_L5_F_STD}} |

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
python run_all_4_hgrn2.py --models distilbert,roberta,gpt2
```

---