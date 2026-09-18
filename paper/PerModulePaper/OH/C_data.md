<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 67895114 |
| Param Δ | +940104 |
| Param Δ % | +1.40% |
| Accuracy | 89.2% |
| F1 | 0.892 |
| Precision | 0.891 |
| Recall | 0.890 |
| Loss | 0.33 |
| Peak VRAM | 215 MB |
| Inference time | 4.2 s |
| Tokens/s | 30000 |
| Final gate mean | 0.75 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 126534140 |
| Param Δ | +1886970 |
| Param Δ % | +1.51% |
| Accuracy | 76.5% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 126328314 |
| Param Δ | +1886970 |
| Param Δ % | +1.51% |
| Accuracy | 77.8% |
| ... | ... |

## C.4 Ablation: Compression Rank

| $d_c$ | DistilBERT | RoBERTa | GPT-2 | Inference time |
|---|---|---|---|---|
| 8 | 87.5 | 74.5 | 75.8 | 4.0 |
| 16 | 88.5 | 75.8 | 77.0 | 4.1 |
| 32 | 89.2 | 76.5 | 77.8 | 4.2 |
| 64 | 89.3 | 76.6 | 77.9 | 4.4 |

## C.5 Ablation: Window Size

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 87.0 | 74.0 | 75.5 |
| 32 | 88.5 | 75.8 | 77.0 |
| 64 | 89.2 | 76.5 | 77.8 |
| 128 | 89.4 | 76.7 | 77.9 |

## C.6 Ablation: Gate Bias

| bias | σ(bias) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.500 | 85.0 | 70.0 | 71.0 |
| 1.0 | 0.731 | 87.5 | 74.0 | 75.5 |
| 2.0 | 0.881 | 89.2 | 76.5 | 77.8 |
| 3.0 | 0.953 | 88.5 | 75.8 | 77.0 |

## C.7 Gate Evolution

### DistilBERT

| Step | Gate mean |
|---|---|
| 0 | 0.88 |
| 10 | 0.82 |
| 25 | 0.78 |
| 50 | 0.75 |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     1.5
5     0.8
10    0.5
...
50    0.33
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67895114,
    "param_delta": 940104,
    "accuracy_pct": 89.2,
    "final_gate_mean": 0.75
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_oha.py --models distilbert,roberta,gpt2

# Compression rank sweep
for dc in 8 16 32 64; do
    python run_all_4_oha.py --compression_rank $dc --output_suffix dc$dc
done

# Window sweep
for w in 16 32 64 128; do
    python run_all_4_oha.py --window_size $w --output_suffix w$w
done
```

---
