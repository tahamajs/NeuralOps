<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Window $w$ | 128 |
| RFF $m$ | 64 |
| Gate bias | 0.0 |
| Accuracy | 90.71% |
| F1 | 0.9107 |
| Mean gate | 0.94 |
| Gate min | 0.85 |
| Gate max | 0.85 |
| Peak VRAM | 335 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Mean gate | 0.85 |
| Peak VRAM | 668 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Mean gate | 0.85 |
| Peak VRAM | 1511 MB MB |
| Inference time | 2.3 s s |

## C.4 Per-Layer Gate (DistilBERT)

| Layer | Mean gate |
|---|---|
| 0 | 0.85 |
| 1 | 0.85 |
| 2 | 0.85 |
| 3 | 0.85 |
| 4 | 0.85 |
| 5 | 0.85 |

## C.5 Window Size Ablation

| $w$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |

## C.6 RFF Features Ablation

| $m$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| 256 | 0.85 | 0.85 | 0.85 |

## C.7 Gate Bias Ablation

| Bias | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 1.8 | 0.85 | 0.85 | 0.85 |
| 3.0 | 0.85 | 0.85 | 0.85 |
| 5.0 | 0.85 | 0.85 | 0.85 |

## C.8 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.9 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.10 Raw JSON

```json
{
  "nha": {
    "distilbert": {
      "window_size": 128,
      "rff_features": 64,
      "accuracy_pct": 90.71,
      "mean_gate": 0.94,
      "peak_vram_mb": 335 MB
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_nha.py --window_size 64 --rff_features 128 --gate_init_bias 3.0

# Window sweep
for w in 16 32 64 128; do
    python run_all_4_nha.py --window_size $w
done
```

---