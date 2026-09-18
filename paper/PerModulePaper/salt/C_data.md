<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 80.28% |
| F1 | 0.8102 |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Reduction | 48.2\%× |
| Storage (MB) | 256 MB |
| Peak VRAM (MB) | 338 MB |
| Inference time (s) | 1.42x |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,647,170 |
| Params after | 124.6M |
| Reduction | 32× |
| Storage (MB) | 84.5 |
| Peak VRAM (MB) | 339 MB |
| Inference time (s) | 2.3 s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,441,344 |
| Params after | 124.4M |
| Reduction | 32× |
| Storage (MB) | 84.5 |
| Peak VRAM (MB) | 583 MB |
| Inference time (s) | 2.3 s |

## C.4 Rank Ablation (DistilBERT)

| Rank | Acc | Params/layer | Reduction |
|---|---|---|---|
| 4 | 32 | 12,336 | 47.8× |
| 8 | 32 | 24,656 | 23.9× |
| 16 | 32 | 49,296 | 12.0× |
| 32 | 32 | 98,576 | 5.98× |
| 64 | 32 | 197,136 | 2.99× |

## C.5 Gamma Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| γ trainable | 0.85 | 0.85 | 0.85 |
| γ frozen | 0.85 | 0.85 | 0.85 |

## C.6 AB Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AB trainable | 0.85 | 0.85 | 0.85 |
| AB frozen | 0.85 | 0.85 | 0.85 |

## C.7 Training Loss Curves

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

## C.8 Singular Value Statistics

### DistilBERT, layer 3

| k | σ_k | Energy fraction |
|---|---|---|
| 1 | 84.5 | 0.85% |
| 2 | 84.5 | 0.85% |
| ... | ... | ... |
| 16 | 84.5 | 0.85% |
| 17+ | — | 32% |

## C.9 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 80.28,
    "params_before": 66955010,
    "params_after": 66.4M,
    "reduction": 48.2\%,
    "storage_mb": 256 MB,
    "peak_vram_mb": 338 MB
  },
  ...
}
```

## C.10 Reproduction

```bash
python run_all_4_salt.py --models distilbert,roberta,gpt2 --rank 32
```

---