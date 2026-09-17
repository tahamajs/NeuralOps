# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 64896000 |
| Attention params before | 2,359,296 |
| Attention params after | 983040 |
| Attention ratio | 2.4× |
| Accuracy | 87.96% |
| F1 | 0.8855 |
| Peak VRAM | 290 MB MB |
| Inference time | 2.1 s |
| Avg rank | 160 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 120500000 |
| Accuracy | 77.64% |
| F1 | 0.7730 |
| Peak VRAM | 586 MB MB |
| Avg rank | 160 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 120300000 |
| Accuracy | 76.38% |
| F1 | 0.8034 |
| Peak VRAM | 302 MB MB |
| Avg rank | 160 |

## C.4 Per-Layer Rank

| Layer | Q r* | K r* | V r* | O r* |
|---|---|---|---|---|
| 0 | 128 | 128 | 192 | 192 |
| 1 | 128 | 128 | 192 | 192 |
| 2 | 128 | 128 | 192 | 192 |
| 3 | 128 | 128 | 192 | 192 |
| 4 | 128 | 128 | 192 | 192 |
| 5 | 128 | 128 | 192 | 192 |

## C.5 Energy τ Ablation

| τ | DistilBERT | Avg Rank | Ratio |
|---|---|---|---|
| 0.80 | 82.0 | 80 | 3.8× |
| 0.85 | 86.0 | 112 | 2.9× |
| 0.92 | 88.0 | 160 | 2.4× |
| 0.95 | 88.8 | 224 | 1.6× |
| 0.99 | 90.5 | 384 | 1.0× |

## C.6 Residual Rank Ablation

| $r_a$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 82.0 | 69.0 | 71.0 |
| 8 | 86.5 | 73.5 | 75.5 |
| 16 | 88.0 | 75.5 | 77.0 |
| 32 | 88.2 | 75.7 | 77.2 |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     0.35
5     0.22
10    0.15
20    0.10
30    0.07
40    0.05
50    0.04
```

## C.8 Reproduction Commands

```bash
python run_all_4_adaptive_svd.py --models distilbert,roberta,gpt2
```

---