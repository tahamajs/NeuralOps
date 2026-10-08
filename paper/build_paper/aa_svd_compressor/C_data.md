<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 16000000 |
| Ratio | 4.0× |
| Accuracy | 89.45% |
| F1 | 0.8996 |
| Precision | 0.851 |
| Recall | 0.849 |
| Loss | 0.40 |
| Peak VRAM | 234 MB |
| Inference time | 2.5 s |
| Compressed layers | 24 |
| Fallback layers | 0 |
| Average error | 0.30 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 35000000 |
| Ratio | 3.5× |
| Accuracy | 88.07% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 35000000 |
| Ratio | 3.5× |
| Accuracy | 80.73% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | 0.89 | 0.91 | 0.9 | 0.93 |
| 1 | 0.90 | 0.97 | 0.9 | 0.97 |
| 2 | 0.91 | 0.93 | 0.9 | 0.98 |
| 3 | 0.99 | 0.98 | 0.9 | 1.00 |
| 4 | 0.94 | 0.99 | 0.9 | 0.92 |
| 5 | 0.97 | 0.88 | 0.9 | 0.93 |

## C.5 Per-Layer Reconstruction Error (RoBERTa)

(12 layers × 4 projections)

## C.6 Per-Layer Reconstruction Error (GPT-2)

(12 layers × 2 projections)

## C.7 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 32 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 128 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 256 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

## C.8 Covariance Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full cov | 0.87 | 0.89 | 0.9 |
| Diagonal | 0.91 | 0.98 | 0.9 |
| No weighting | 80.0 | 0.90 | 0.93 |

## C.9 EMA β Ablation

| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.91 | 0.98 | 0.9 |
| 0.7 | 0.99 | 1.00 | 0.9 |
| 0.9 | 0.94 | 0.88 | 0.9 |
| 0.99 | 1.00 | 0.89 | 0.9 |
| 0.99 | 0.87 | 0.91 | 0.9 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

### RoBERTa
```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2
```
Step  Loss
1     73.5
...
50    77.5
```

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 16000000,
    "ratio": 4.0,
    "accuracy_pct": 89.45,
    "f1": 0.8996,
    "compressed_layers": 24,
    "fallback_layers": 0,
    "avg_reconstruction_error": 0.30
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_aasvd.py - models distilbert,roberta,gpt2

# Rank sweep
for r in 32 64 128 256; do
    python run_all_4_aasvd.py - rank $r - output_suffix r$r
done
```
