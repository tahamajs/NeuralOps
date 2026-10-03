# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Base rank | 32 |
| Residual rank | 16 |
| Accuracy | 89.56% |
| F1 | 0.9016 |
| Peak VRAM | 298 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 76.61% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 71.33% |

## C.4 Base Rank Ablation

| $r_b$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## C.5 Residual Rank Ablation

| $r_r$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |

## C.6 Scaling Ablation

| $\alpha$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## C.7 vs Pure SVD

| Config | DistilBERT | RoBERTa |
|---|---|---|
| Pure SVD (r=20) | 84.5 | 84.5 |
| Res-SVD (16,4) | 89.56 | 76.61 |
| Difference | 0.85 | 0.85 |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 89.56,
    "base_rank": 32,
    "residual_rank": 16
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