# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| KV length (baseline) | 128 |
| KV length (TCA) | 0.85 |
| Attention FLOP reduction | 48.5\%% |
| Peak VRAM | 323 MB MB |
| Inference time | 1.42x s |
| Score net params | 49,985 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Accuracy | 50.92% |
| KV length (TCA) | 0.85 |
| Peak VRAM | 369 MB MB |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Accuracy | 50.92% |
| KV length (TCA) | 0.85 |
| Peak VRAM | 571 MB MB |

## C.4 ρ Ablation

| ρ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.9 | 32 | 32 | 32 |
| 0.7 | 32 | 32 | 32 |
| 0.5 | 32 | 32 | 32 |
| 0.3 | 32 | 32 | 32 |
| 0.1 | 32 | 32 | 32 |

## C.5 Score Net Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| norm + score_net | 0.85 | 0.85 | 0.85 |
| norm only | 0.85 | 0.85 | 0.85 |
| score_net only | 0.85 | 0.85 | 0.85 |
| random | 32 | 32 | 32 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

## C.7 Reproduction Commands

```bash
python run_all_4_token_compressed.py --compression_ratio 0.5
```

---