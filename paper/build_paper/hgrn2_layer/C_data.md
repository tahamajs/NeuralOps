# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Trainable | 0.85 |
| Accuracy | 62.96% |
| F1 | 0.7103 |
| Peak VRAM | 447 MB MB |
| Time (L=128) | 0.85 s |
| Time (L=1024) | 0.85 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 52.06% |
| F1 | 0.6645 |
| Peak VRAM | 788 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 71.44% |
| F1 | 0.7255 |
| Peak VRAM | 507 MB MB |

## C.4 Speed vs L

| L | Baseline | Ours |
|---|---|---|
| 128 | 1.90 s | 0.85 s |
| 256 | 3.50 s | 0.85 s |
| 512 | 12.0 s | 0.85 s |
| 1024 | 45.0 s | 0.85 s |
| 2048 | 180.0 s | 0.85 s |
| 4096 | 720.0 s | 0.85 s |

## C.5 Chunk Size Ablation

| Chunk | DistilBERT | Time (L=512) |
|---|---|---|
| 8 | 0.85 | 0.85 s |
| 16 | 0.85 | 0.85 s |
| 32 | 0.85 | 0.85 s |
| 64 | 0.85 | 0.85 s |

## C.6 Forget Gate Statistics

| Layer | Mean f | Std f |
|---|---|---|
| 0 | 0.85 | 84.5 |
| 3 | 0.85 | 84.5 |
| 5 | 0.85 | 84.5 |

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
python run_all_4_hgrn2.py --models distilbert,roberta,gpt2
```

---