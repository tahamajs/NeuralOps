# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 68.2 |
| Change | 1.8% |
| Accuracy | 83.72% |
| F1 | 0.8515 |
| Peak VRAM | 212 MB |
| Inference time | 2.2 s |
| SVD energy | 0.95 |
| Max rank | 384 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 127.0 |
| Accuracy | 86.47% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 127.0 |
| Accuracy | 69.50% |

## C.4 Rank Ablation

| max_rank | DistilBERT | Params |
|---|---|---|
| 32 | 78.0 | 52.0M |
| 64 | 83.0 | 55.0M |
| 128 | 87.0 | 59.0M |
| 256 | 89.0 | 64.0M |
| 384 | 89.5 | 68.2M |

## C.5 Head Split Ablation

| Split | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| All local | 85.0 | 72.0 | 74.0 |
| 8/4 | 89.5 | 76.5 | 78.5 |
| 6/6 | 88.5 | 75.5 | 77.5 |
| All global | 82.0 | 68.0 | 70.0 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    89.5
```

## C.7 Reproduction Commands

```bash
python run_all_4_aegis_v5.py - svd_energy 0.95 - max_rank 128
```

---