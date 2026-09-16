# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Base rank | {{BASE_RANK}} |
| Residual rank | {{RESIDUAL_RANK}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |

## C.4 Base Rank Ablation

| $r_b$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_B8}} | {{R_B8}} | {{G_B8}} |
| 16 | {{D_B16}} | {{R_B16}} | {{G_B16}} |
| 32 | {{D_B32}} | {{R_B32}} | {{G_B32}} |
| 64 | {{D_B64}} | {{R_B64}} | {{G_B64}} |

## C.5 Residual Rank Ablation

| $r_r$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_R0}} | {{R_R0}} | {{G_R0}} |
| 2 | {{D_R2}} | {{R_R2}} | {{G_R2}} |
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |

## C.6 Scaling Ablation

| $\alpha$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_A05}} | {{R_A05}} | {{G_A05}} |
| 1.0 | {{D_A10}} | {{R_A10}} | {{G_A10}} |
| 2.0 | {{D_A20}} | {{R_A20}} | {{G_A20}} |

## C.7 vs Pure SVD

| Config | DistilBERT | RoBERTa |
|---|---|---|
| Pure SVD (r=20) | {{D_SVD20}} | {{R_SVD20}} |
| Res-SVD (16,4) | {{D_ACC}} | {{R_ACC}} |
| Difference | {{D_DIFF}} | {{R_DIFF}} |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "base_rank": {{BASE_RANK}},
    "residual_rank": {{RESIDUAL_RANK}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_res_svd.py --models distilbert,roberta,gpt2

# Base rank sweep
for rb in 8 16 32 64; do
    python run_all_4_res_svd.py --base_rank $rb --output_suffix rb$rb
done

# Residual rank sweep
for rr in 0 2 4 8 16; do
    python run_all_4_res_svd.py --residual_rank $rr --output_suffix rr$rr
done
```

---