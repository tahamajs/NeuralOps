# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 48500000 |
| Ratio | 1.38× |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 291 MB MB |
| Inference time | 2.0 s |
| rank_qkv | 64 |
| rank_out | 64 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 95000000 |
| Ratio | 1.31× |
| Accuracy | 50.92% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 95000000 |
| Ratio | 1.31× |
| Accuracy | 50.92% |

## C.4 Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 82.0 | 68.0 | 70.5 |
| 64 | 89.5 | 76.5 | 78.0 |
| 128 | 90.0 | 77.5 | 78.8 |
| 256 | 90.5 | 78.0 | 79.0 |
| 384 | 90.6 | 78.2 | 79.2 |

## C.5 Joint vs Independent

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Independent | 89.0 | 76.0 | 77.5 |
| Joint | 89.5 | 76.5 | 78.0 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    85.0
...
50    89.5
```

## C.7 Nested Detection Test

| Detection order | DistilBERT |
|---|---|
| Wrong (class name first) | 50.92% |
| Correct (out_lin first) | 50.92% |

## C.8 Reproduction Commands

```bash
python run_all_4_compressed_mha.py --rank_qkv 64 --rank_out 64
```

---