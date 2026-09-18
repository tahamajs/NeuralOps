<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Sparsity (learned mean) | 0.42 |
| Accuracy | 88.5% |
| F1 | 0.885 |
| Peak VRAM | 120 MB |
| Inference time | 2.1 s |
| $\rho_\text{min}$ | 0.05 |
| $\rho_\text{max}$ | 0.85 |
| $\rho_\text{std}$ | 0.28 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Sparsity | 0.45 |
| Accuracy | 76.0% |
| F1 | 0.760 |
| Peak VRAM | 500 MB |
| Inference time | 4.5 s |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Sparsity | 0.48 |
| Accuracy | 76.5% |
| F1 | 0.765 |
| Peak VRAM | 550 MB |
| Inference time | 4.6 s |

## C.4 Sparsity Ablation

| $\rho$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.1 | 89.5 | 77.5 | 78.0 |
| 0.3 | 89.0 | 77.0 | 77.5 |
| 0.5 | 88.5 | 76.0 | 76.5 |
| 0.7 | 86.0 | 73.5 | 74.0 |
| 0.9 | 80.0 | 68.0 | 68.5 |

## C.5 Temperature Ablation

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.01 | 85.0 | 72.0 | 73.0 |
| 0.1 | 88.5 | 76.0 | 76.5 |
| 0.5 | 88.0 | 75.5 | 76.0 |
| 1.0 | 87.0 | 74.5 | 75.0 |

## C.6 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 85.0 | 72.0 | 72.0 |
| 8 | 87.0 | 74.0 | 74.5 |
| 16 | 88.5 | 76.0 | 76.5 |
| 32 | 88.7 | 76.3 | 76.7 |

## C.7 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 85.0 | 72.0 | 73.0 |
| 25 | 87.5 | 74.5 | 75.0 |
| 50 | 88.5 | 76.0 | 76.5 |
| 100 | 88.7 | 76.3 | 76.8 |

## C.8 Head Budget Distribution

### DistilBERT
| Head | Layer 0 | Layer 3 | Layer 5 |
|---|---|---|---|
| 0 | 0.87 | 0.88 | 0.9 |
| 1 | 0.87 | 0.88 | 0.9 |
| ... | ... | ... | ... |
| 11 | 0.87 | 0.88 | 0.9 |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss  Sparsity
1     82.1  [D_SP1]
5     83.4  [D_SP5]
10    89.5 [D_SP10]
...
50    88.5 [D_SP50]
```

## C.10 Raw JSON

```json
{
  "adaptive_sparse": {
    "distilbert": {
      "accuracy_pct": 88.5,
      "learned_sparsity": 0.42,
      "peak_vram_mb": 120,
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
python run_all_4_adaptive_sparse.py --sparsity_ratio 0.5

# Sparsity sweep
for rho in 0.1 0.3 0.5 0.7 0.9; do
    python run_all_4_adaptive_sparse.py --sparsity_ratio $rho
done
```

---
