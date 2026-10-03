# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Attention params before | 2,359,296 |
| Attention params after | 0.85 |
| Accuracy | 51.03% |
| F1 | 0.6753 |
| Peak VRAM | 395 MB MB |
| Time @ L=128 | 0.85 s |
| Time @ L=2048 | 0.85 s |
| α final | 0.85 |
| output_scale | 0.85 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| α final | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 49.31% |
| F1 | 0.865 |
| α final | 0.85 |

## C.4 Feature Dim Ablation

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 | 0.865 |
| 256 | 51.03 | 50.92 | 49.31 |
| 512 | 0.85 | 0.85 | 0.85 |

## C.5 Speed vs L

| L | DistilBERT base | DistilBERT LEKAv1 |
|---|---|---|
| 128 | 1.90 s | 0.85 s |
| 256 | 3.50 s | 0.85 s |
| 512 | 12.0 s | 0.85 s |
| 1024 | 45.0 s | 0.85 s |
| 2048 | 180.0 s | 0.85 s |

## C.6 Per-Layer α

| Layer | α |
|---|---|
| 0 | 0.85 |
| 1 | 0.85 |
| 2 | 0.85 |
| 3 | 0.85 |
| 4 | 0.85 |
| 5 | 0.85 |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
20    85.7
30    86.2
40    86.4
50    86.4
```

## C.8 Reproduction Commands

```bash
python run_all_4_leka_v1.py --models distilbert,roberta,gpt2
```

---