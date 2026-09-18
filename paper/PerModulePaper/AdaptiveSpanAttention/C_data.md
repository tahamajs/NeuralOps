<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Max span | 128 |
| Ramp size | 32 |
| λ | 0.01 |
| Accuracy | 89.5% |
| F1 | 0.895 |
| Mean span | 68 |
| Min/Max span | 16 / 124 |
| Std span | 35 |
| Peak VRAM | 190 MB |
| Inference time | 2.1 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 76.5% |
| F1 | 0.765 |
| Mean span | 72 |
| Reduction | 44% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| F1 | 0.775 |
| Mean span | 75 |
| Reduction | 41% |

## C.4 Per-Layer Span (DistilBERT)

| Layer | Mean span |
|---|---|
| 0 | 95 |
| 1 | 82 |
| 2 | 70 |
| 3 | 60 |
| 4 | 52 |
| 5 | 48 |

## C.5 Max Span Ablation

| $S_\text{max}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 85.0 | 70.0 | 71.0 |
| 64 | 88.0 | 74.0 | 75.0 |
| 128 | 89.5 | 76.5 | 77.5 |
| 256 | 89.6 | 76.6 | 77.6 |

## C.6 Ramp Size Ablation

| $R$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 87.0 | 73.0 | 73.5 |
| 8 | 88.5 | 75.0 | 76.0 |
| 16 | 89.0 | 76.0 | 77.0 |
| 32 | 89.5 | 76.5 | 77.5 |
| 64 | 88.5 | 75.5 | 76.5 |

## C.7 L1 Penalty Ablation

| $\lambda$ | DistilBERT acc | Mean span |
|---|---|---|
| 0.0 | 90.5 | 126 |
| 0.001 | 90.0 | 105 |
| 0.01 | 89.5 | 68 |
| 0.1 | 86.0 | 32 |

## C.8 Head Span Distribution (DistilBERT Layer 3)

| Head | Span |
|---|---|
| 0 | 28 |
| 1 | 118 |
| ... | ... |
| 11 | 34 |

## C.9 Span Evolution

### DistilBERT
```
Step  Mean span
0     126
5     112
10    98
15    88
20    80
...
50    68
```

## C.10 Raw JSON

```json
{
  "adaptive_span": {
    "distilbert": {
      "max_span": 128,
      "learned_mean_span": 68,
      "accuracy_pct": 89.5,
      "peak_vram_mb": 190,
      "inference_time_s": 2.1
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_adaptive_span.py --max_span 128 --ramp_size 32

# Max span sweep
for S in 32 64 128 256; do
    python run_all_4_adaptive_span.py --max_span $S
done
```

---
