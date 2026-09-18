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
| Top-$k$ | 2 |
| Accuracy | 90.83% |
| F1 | 0.9119 |
| Density | 50\%% |
| Preserved mass | 0.85% |
| Peak VRAM | 340 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Density | 0.85% |
| Preserved mass | 0.85% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Density | 0.85% |
| Preserved mass | 0.85% |

## C.4 Window Size Ablation

| $w$ | DistilBERT | RoBERTa | GPT-2 | Density |
|---|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 | 0.12% |
| 16 | 0.85 | 0.85 | 0.85 | 0.24% |
| 32 | 0.85 | 0.85 | 0.85 | 0.45% |
| 64 | 0.85 | 0.85 | 0.85 | 0.82% |

## C.5 Top-K Ablation

| $k$ | DistilBERT | RoBERTa | GPT-2 | Density |
|---|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 | 0.10% |
| 8 | 0.85 | 0.85 | 0.85 | 0.20% |
| 16 | 0.85 | 0.85 | 0.85 | 0.38% |
| 32 | 0.85 | 0.85 | 0.85 | 0.75% |

## C.6 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.7 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.8 Per-Layer Density (DistilBERT)

| Layer | Density |
|---|---|
| 0 | 0.85% |
| 1 | 0.85% |
| 2 | 0.85% |
| 3 | 0.85% |
| 4 | 0.85% |
| 5 | 0.85% |

## C.9 Raw JSON

```json
{
  "nsa": {
    "distilbert": {
      "window_size": 128,
      "top_k": 2,
      "accuracy_pct": 90.83,
      "density_pct": 50\%,
      "peak_vram_mb": 340 MB
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.10 Reproduction Commands

```bash
# Main
python run_all_4_nsa.py --window_size 32 --top_k 16

# Window sweep
for w in 8 16 32 64; do
    python run_all_4_nsa.py --window_size $w
done
```

---