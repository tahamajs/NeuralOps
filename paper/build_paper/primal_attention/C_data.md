<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Landmarks $m$ | 64 |
| Pinv iters $K$ | 4 |
| Accuracy | 90.14% |
| F1 | 0.9034 |
| Attention memory | 12 KB |
| Inference time | 2.5 s |
| Peak VRAM | 668 MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 68.00% |
| F1 | 0.6767 |
| Attention memory | 12 KB |
| Inference time | 5.3 s |
| Peak VRAM | 1442 MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 63.07% |
| F1 | 0.6349 |
| Attention memory | 12 KB |
| Inference time | 5.5 s |
| Peak VRAM | 1685 MB |

## C.4 Landmark Ablation

| $m$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 78.0 | 65.0 | 67.0 |
| 32 | 83.5 | 70.0 | 71.5 |
| 64 | 86.5 | 74.0 | 75.0 |
| 128 | 88.0 | 75.5 | 76.5 |
| 256 | 88.5 | 76.0 | 77.0 |

## C.5 Pinv Iterations Ablation

| $K$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 80.0 | 67.0 | 68.0 |
| 2 | 84.0 | 71.0 | 72.0 |
| 4 | 86.5 | 74.0 | 75.0 |
| 6 | 86.7 | 74.2 | 75.2 |
| 8 | 86.8 | 74.3 | 75.3 |

## C.6 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 83.0 | 70.0 | 71.0 |
| 8 | 85.0 | 72.0 | 73.0 |
| 16 | 86.5 | 74.0 | 75.0 |
| 32 | 86.7 | 74.5 | 75.3 |

## C.7 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 82.0 | 68.0 | 69.0 |
| 25 | 85.0 | 72.0 | 73.0 |
| 50 | 86.5 | 74.0 | 75.0 |
| 100 | 86.8 | 74.5 | 75.3 |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    82.0
...
50    86.5
```

### RoBERTa
```
Step  Loss
1     74.2
...
50    74.0
```

### GPT-2
```
Step  Loss
1     73.5
...
50    75.0
```

## C.9 Raw JSON

```json
{
  "primal": {
    "distilbert": {
      "num_landmarks": 64,
      "pinv_iters": 4,
      "accuracy_pct": 90.14,
      "attention_mem_bytes": 12288,
      "peak_vram_mb": 668 MB,
      "inference_time_s": 2.5
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.10 Reproduction Commands

```bash
# Main
python run_all_4_primal.py --rank 64 --pinv_iters 4

# Rank sweep
for m in 16 32 64 128 256; do
    python run_all_4_primal.py --rank $m --output_suffix m$m
done
```

---
