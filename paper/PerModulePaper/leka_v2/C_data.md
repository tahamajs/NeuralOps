# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Attention params | 0.85 |
| Accuracy | 86.01% |
| F1 | 0.8688 |
| Landmark ratio | 0.85 |
| Mean local gate | 0.85 |
| Peak VRAM | 369 MB MB |
| Time @ L=128 | 0.85 s |
| Time @ L=1024 | 0.85 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Landmark ratio | 0.85 |
| Mean gate | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Landmark ratio | 0.85 |
| Mean gate | 0.85 |

## C.4 ρ Ablation

| ρ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.25 | 32 | 32 | 32 |
| 0.50 | 86.01 | 50.92 | 50.92 |
| 0.75 | 32 | 32 | 32 |
| 1.00 | 32 | 32 | 32 |

## C.5 Feature Dim

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 86.01 | 50.92 | 50.92 |
| 256 | 0.85 | 0.85 | 0.85 |
| 512 | 0.85 | 0.85 | 0.85 |

## C.6 Window Size

| W | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 128 | 86.01 | 50.92 | 50.92 |
| 256 | 0.85 | 0.85 | 0.85 |

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
python run_all_4_leka_v2.py --models distilbert,roberta,gpt2
```

---