<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Accuracy | 81.88% |
| F1 | 0.8260 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 289 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |
| Avg rank | 32 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 339 MB MB |
| Compressed layers | 0.85 |
| Avg rank | 32 |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 478 MB MB |
| Compressed layers | 0.85 |
| Avg rank | 32 |

## C.4 Effective Rank Distribution (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 | 32 |
| 3 | 32 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 | 32 |
| 5 | 32 | 32 | 32 | 32 |

## C.5 Effective Rank Distribution (RoBERTa)

(12 layers × 4 projections)

## C.6 Effective Rank Distribution (GPT-2)

(12 layers × 4 projections)

## C.7 Singular Value Statistics

**DistilBERT, layer 0:**

| Projection | σ_1 | σ_10 | σ_32 | σ_64 |
|---|---|---|---|---|
| Q | 84.5 | 84.5 | 84.5 | 84.5 |
| K | 84.5 | 84.5 | 84.5 | 84.5 |
| V | 84.5 | 84.5 | 84.5 | 84.5 |
| O | 84.5 | 84.5 | 84.5 | 84.5 |

## C.8 Reconstruction Error (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.9 Rank Cap Ablation

| $r_\text{max}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 32 | 2.1\times× | 32 | 32 |
| 64 | 32 | 2.1\times× | 32 | 32 |
| 128 | 32 | 2.1\times× | 32 | 32 |
| 192 | 32 | 2.1\times× | 32 | 32 |

## C.10 Energy Threshold Ablation

| $\eta$ | DistilBERT | Avg rank | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.95 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.99 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.999 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.11 LoRA Rank Ablation

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.12 Explicit vs Symmetric

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Explicit | 81.88 | 50.92 | 50.92 |
| Symmetric | 84.5 | 84.5 | 84.5 |

## C.13 Training Loss Curves

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

## C.14 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 81.88,
    "f1": 0.8260,
    "compressed_layers": 2.0x,
    "avg_rank": 32
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---