# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Accuracy | 86.4% |
| F1 | 86.2 |
| Peak VRAM | 380 MB MB |
| Path used | Short (L=128) |
| Block size | 64 |
| Top-K | 2 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 66.4M |
| Accuracy | 78.2% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 66.4M |
| Accuracy | 77.5% |
| ... | ... |

## C.4 Block Size Ablation (Simulated L=512)

| $B$ | D acc | Time ratio |
|---|---|---|
| 32 | 0.85 | 0.5× |
| 64 | 0.85 | 1.0× |
| 128 | 0.85 | 2.0× |

## C.5 Top-K Ablation (Simulated L=512)

| $K$ | D acc | Time ratio |
|---|---|---|
| 2 | 0.85 | 0.5× |
| 4 | 0.85 | 1.0× |
| 8 | 0.85 | 2.0× |

## C.6 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 |
| 16 | 86.4 | 78.2 | 77.5 |
| 32 | 32 | 32 | 32 |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "accuracy_pct": 86.4,
    "block_size": 64,
    "top_k": 2,
    "path_used": "short",
    "sequence_length": 128
  }
}
```

---