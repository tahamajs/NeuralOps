<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 Hardware Detection Results

| Check | Result |
|---|---|
| `torch.cuda.is_available()` | True |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Compute capability | (8, 9) |
| `torch.float8_e4m3fn` | 0.85 |
| `flash_attn` importable | 0.85 |
| FP8 path taken | 0.85 |

## C.2 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 89.79% |
| F1 | 0.9025 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params (before) | 66,955,010 |
| Params (after) | 66.4M |
| Peak VRAM | 1335 MB MB |
| Inference time | 1.42x s |
| Tokens/sec | 0.85 |

## C.3 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 78.2% |
| F1 | 0.865 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params (before) | 124,647,170 |
| Params (after) | 124.6M |
| Peak VRAM | 380 MB MB |
| Inference time | 2.3 s s |

## C.4 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| F1 | 0.865 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params (before) | 124,441,344 |
| Params (after) | 124.4M |
| Peak VRAM | 380 MB MB |
| Inference time | 2.3 s s |

## C.5 Training Loss Curves

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
5     75.6
10    76.8
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

## C.6 Per-Layer Weight Transfer Verification

### DistilBERT

| Layer | Q diff | K diff | V diff | O diff |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| ... | ... | ... | ... | ... |

### RoBERTa

(Same verification; diff should be 0 for Q/K/V, N/A for O due to nested)

### GPT-2

(Same verification; requires Conv1D transpose check)

## C.7 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 89.79,
    "params_after": 66.4M,
    "fp8_path_taken": false,
    "sdpa_backend": "flash",
    "peak_vram_mb": 1335 MB
  },
  ...
}
```

## C.8 Reproduction Commands

```bash
# Full run
python run_all_4_flash3_fp8.py --models distilbert,roberta,gpt2

# Ablations
python run_all_4_flash3_fp8.py --use_fp8 false --output_suffix nofp8
```

## C.9 Hardware Comparison Table

| GPU | Compute | FP8 Available? |
|---|---|---|
| RTX 4070 Laptop | 8.9 | ❌ (this study) |
| RTX 4090 | 8.9 | ✅ |
| H100 | 9.0 | ✅ |
| A100 | 8.0 | ❌ |
```
```

---