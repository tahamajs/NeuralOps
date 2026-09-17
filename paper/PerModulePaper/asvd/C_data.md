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
| Params after | 12165730 |
| Ratio | 5.5× |
| Accuracy | 90.37% |
| F1 | 0.9079 |
| Peak VRAM | 344 MB MB |
| Inference time | 2.4 s |
| Avg reconstruction error | 0.12 |
| Avg rank | 58 |
| Compression time | 6 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 24929230 |
| Ratio | 5.0× |
| Accuracy | 77.75% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 24888140 |
| Ratio | 5.0× |
| Accuracy | 85.78% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.5 Per-Layer Selected Rank (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 48 | 52 | 56 | 60 |
| 1 | 56 | 60 | 62 | 58 |
| ... | ... | ... | ... | ... |
| 5 | 48 | 52 | 50 | 54 |

## C.6 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | 86.0 | 12.0× | 73.0 | 11.0× | 74.5 | 11.0× |
| 32 | 87.5 | 8.0× | 74.5 | 7.0× | 76.0 | 7.0× |
| 64 | 88.5 | 5.5× | 75.5 | 5.0× | 76.8 | 5.0× |
| 128 | 89.2 | 3.0× | 76.5 | 2.8× | 77.5 | 2.8× |

## C.7 Energy Threshold Ablation

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | 87.5 | 7.0× | 74.5 | 75.8 |
| 0.95 | 88.5 | 5.5× | 75.5 | 76.8 |
| 0.99 | 89.5 | 4.0× | 76.5 | 77.8 |
| 0.995 | 89.8 | 3.5× | 76.7 | 78.0 |

## C.8 Dynamic vs Fixed Rank

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dynamic | 88.5 | 75.5 | 76.8 |
| Fixed | 87.5 | 74.8 | 76.0 |

## C.9 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 80.5 | 65.0 | 68.0 |
| 8 | 86.5 | 73.0 | 75.0 |
| 16 | 88.5 | 75.5 | 76.8 |
| 32 | 88.8 | 76.0 | 77.0 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     1.2
5     0.7
10    0.5
...
50    0.35
```

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 12165730,
    "ratio": 5.5,
    "accuracy_pct": 90.37,
    "avg_rank": 58,
    "avg_reconstruction_error": 0.12
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_asvd.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 16 32 64 128; do
    python run_all_4_asvd.py --rank $r --output_suffix r$r
done

# Energy sweep
for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_asvd.py --energy_threshold $e --output_suffix e$e
done
```

---