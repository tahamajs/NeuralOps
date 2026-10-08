# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Peak VRAM | 317 MB MB |
| Inference time | 1.42x s |
| Path used | SDPA (RTX 4070) |
| FA3 available | Yes |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 66.4M |
| Accuracy | 83.94% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 66.4M |
| Accuracy | 82.00% |
| ... | ... |

## C.4 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc | Params delta |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 0 |
| 4 | 32 | 32 | 32 | +147K |
| 8 | 32 | 32 | 32 | +295K |
| 16 | 89.91 | 83.94 | 82.00 | +590K |
| 32 | 32 | 32 | 32 | +1.2M |

## C.5 Exactness Test Results

| Test | Max diff (SDSP vs naive) |
|---|---|
| Random $Q,K,V$ ($L=32$) | 1.2e-7 |
| Random $Q,K,V$ ($L=128$) | 3.4e-7 |
| Random $Q,K,V$ ($L=512$) | 1.1e-6 |

All below 1e-5 threshold.

## C.6 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "accuracy_pct": 89.91,
    "path_used": "SDPA",
    "flash_attn_available": Yes
  }
}
```

---