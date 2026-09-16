# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_ABS}} |
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| rank_qkv | {{RANK_QKV}} |
| rank_out | {{RANK_OUT}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS_ABS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS_ABS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |

## C.4 Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{R_R128}} | {{G_R128}} |
| 256 | {{D_R256}} | {{R_R256}} | {{G_R256}} |
| 384 | {{D_R384}} | {{R_R384}} | {{G_R384}} |

## C.5 Joint vs Independent

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Independent | {{D_INDEP}} | {{R_INDEP}} | {{G_INDEP}} |
| Joint | {{D_JOINT}} | {{R_JOINT}} | {{G_JOINT}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

## C.7 Nested Detection Test

| Detection order | DistilBERT |
|---|---|
| Wrong (class name first) | 50.92% |
| Correct (out_lin first) | {{D_ACC}}% |

## C.8 Reproduction Commands

```bash
python run_all_4_compressed_mha.py --rank_qkv 64 --rank_out 64
```
```

---