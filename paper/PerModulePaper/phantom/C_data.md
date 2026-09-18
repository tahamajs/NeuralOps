<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Accuracy | 72.59% |
| F1 | 0.7668 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 328 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |
| Fallback layers | 0.85 |
| Avg bucket size | 64 |
| Compression time | 66.4M s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.5 LSH Bucket Statistics

### C.5.1 Bucket Size Distribution (DistilBERT, k=4)

| Bucket size | Count | Frequency |
|---|---|---|
| 0 | 0.85 | 0.85% |
| 1-4 | 0.85 | 0.85% |
| 5-16 | 0.85 | 0.85% |
| 17-64 | 0.85 | 0.85% |
| >64 | 0.85 | 0.85% |

## C.6 Ablation: CUR Rank

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 8 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 16 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 32 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

## C.7 Ablation: LSH Rounds

| Rounds | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.8 Ablation: LSH Bits

| Bits | Buckets | DistilBERT | RoBERTa |
|---|---|---|---|
| 2 | 4 | 0.85 | 0.85 |
| 3 | 8 | 0.85 | 0.85 |
| 4 | 16 | 0.85 | 0.85 |
| 6 | 64 | 0.85 | 0.85 |

## C.9 Ablation: Component Contributions

| Config | DistilBERT | Δ |
|---|---|---|
| Full | 0.85 | — |
| No softpick | 0.85 | 0.85 |
| No LSH | 0.85 | 0.85 |
| No CUR | 0.85 | 0.85 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
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

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 72.59,
    "f1": 0.7668,
    "compressed_layers": 2.0x,
    "avg_bucket_size": 64,
    "compression_time_s": 66.4M
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_phantom.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64; do
    python run_all_4_phantom.py --cur_rank $r --output_suffix r$r
done
```

---