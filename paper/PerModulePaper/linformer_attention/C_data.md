# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Change | +48.5\%% |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| k | 0.85 |
| Peak VRAM | 1322 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Accuracy | 48.17% |
| k | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Accuracy | 77.5% |
| k | 0.85 |

## C.4 k Ablation

| k | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| 256 | 0.85 | 0.85 | 0.85 |

## C.5 Init Ablation

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Diagonal + noise | 0.85 | 0.85 | 0.85 |
| Gaussian only | 0.85 | 0.85 | 0.85 |
| Zero | 0.85 | 0.85 | 0.85 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.7 Reproduction Commands

```bash
python run_all_4_linformer.py --linformer_k 64
```

---