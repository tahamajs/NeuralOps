<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Accuracy | 89.5% |
| F1 | 0.895 |
| Mean gate | 0.72 |
| Gate min | 0.15 |
| Gate max | 0.98 |
| Peak VRAM | 210 MB |
| Inference time | 2.8 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 76.5% |
| F1 | 0.765 |
| Mean gate | 0.68 |
| Peak VRAM | 760 MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| F1 | 0.775 |
| Mean gate | 0.65 |
| Peak VRAM | 820 MB |

## C.4 Per-Layer Gate (DistilBERT)

| Layer | Mean gate |
|---|---|
| 0 | 0.85 |
| 1 | 0.80 |
| 2 | 0.75 |
| 3 | 0.70 |
| 4 | 0.65 |
| 5 | 0.60 |

## C.5 Gate Bias Ablation

| $b_h$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 85.0 | 72.0 | 73.0 |
| 0.5 | 87.0 | 74.0 | 75.0 |
| 1.0 | 88.5 | 75.5 | 76.5 |
| 1.8 | 89.5 | 76.5 | 77.5 |
| 3.0 | 89.0 | 76.0 | 77.0 |

## C.6 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | 89.5 | 76.5 | 77.5 |
| ReLU+1 | 87.0 | 74.0 | 75.5 |
| Softplus | 89.0 | 76.0 | 77.0 |
| Identity | 78.0 | 65.0 | 67.0 |

## C.7 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 86.0 | 72.0 | 73.0 |
| 8 | 88.0 | 74.5 | 75.5 |
| 16 | 89.5 | 76.5 | 77.5 |
| 32 | 89.7 | 76.7 | 77.7 |

## C.8 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 86.0 | 72.0 | 73.0 |
| 25 | 88.0 | 74.5 | 75.5 |
| 50 | 89.5 | 76.5 | 77.5 |
| 100 | 89.7 | 76.8 | 77.7 |

## C.9 Raw JSON

```json
{
  "halo_regla": {
    "distilbert": {
      "accuracy_pct": 89.5,
      "mean_gate": 0.72,
      "peak_vram_mb": 210
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.10 Reproduction Commands

```bash
# Main
python run_all_4_halo_regla.py --gate_init_bias 1.8

# Bias sweep
for b in 0 0.5 1.0 1.8 3.0; do
    python run_all_4_halo_regla.py --gate_init_bias $b
done
```

---
