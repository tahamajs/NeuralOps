# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_ABS}} |
| Trainable | {{D_TRAINABLE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Page size | 64 |
| k_experts | 4 |
| SVD rank | 16 |
| Num pages | {{D_PAGES}} |
| Page-0 rate | {{D_P0_RATE}}% |
| Aux loss | {{D_AUX}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS_ABS}} |
| Accuracy | {{R_ACC}}% |
| Page-0 rate | {{R_P0_RATE}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS_ABS}} |
| Accuracy | {{G_ACC}}% |
| Page-0 rate | {{G_P0_RATE}}% |

## C.4 k_experts Ablation

| k | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_K1}} | {{R_K1}} | {{G_K1}} |
| 2 | {{D_K2}} | {{R_K2}} | {{G_K2}} |
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} |

## C.5 Page Size Ablation

| P | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_P16}} | {{R_P16}} | {{G_P16}} |
| 32 | {{D_P32}} | {{R_P32}} | {{G_P32}} |
| 64 | {{D_P64}} | {{R_P64}} | {{G_P64}} |
| 128 | {{D_P128}} | {{R_P128}} | {{G_P128}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
...
50    {{D_S50}}
```

## C.7 Scaling

| L | Standard | MoE-DeltaKV |
|---|---|---|
| 128 | {{T128_STD}} | {{T128_MOE}} |
| 256 | {{T256_STD}} | {{T256_MOE}} |
| 512 | {{T512_STD}} | {{T512_MOE}} |
| 1024 | {{T1024_STD}} | {{T1024_MOE}} |

## C.8 Reproduction Commands

```bash
python run_all_4_moe_deltakv_paged.py --page_size 64 --k_experts 4
```
```

---