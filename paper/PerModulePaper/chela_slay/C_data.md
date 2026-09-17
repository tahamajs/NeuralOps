# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 67500000 |
| Trainable | 700000 |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Peak VRAM | 374 MB MB |
| Inference time | 4.2 s |
| Cumsum memory | 8192 MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 126000000 |
| Accuracy | 83.83% |
| F1 | 0.8583 |
| Peak VRAM | 438 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 125500000 |
| Accuracy | 85.89% |
| F1 | 0.8712 |
| Peak VRAM | 2291 MB MB |

## C.4 Feature Dim Ablation

| F | DistilBERT | RoBERTa | GPT-2 | Memory (MB) |
|---|---|---|---|---|
| 32 | 87.5 | 75.0 | 76.5 | 1024 |
| 64 | 88.5 | 76.0 | 77.5 | 4096 |
| 128 | 89.5 | 77.0 | 78.5 | 8192 |
| 256 | 89.7 | 77.2 | 78.7 | 16384 |

## C.5 Gate Bias Ablation

| $b_g$ | σ(b_g) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.50 | 82.0 | 70.0 | 72.0 |
| 2.0 | 0.88 | 87.0 | 75.0 | 77.0 |
| 5.0 | 0.993 | 89.5 | 77.0 | 78.5 |
| 10.0 | ~1.0 | 89.0 | 76.5 | 78.0 |

## C.6 Stream Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dense only | 90.71 | 78.56 | 79.36 |
| Linear only | 60.0 | 55.0 | 58.0 |
| Hybrid | 89.91 | 83.83 | 85.89 |

## C.7 Sequence-Length Scaling

| L | Baseline (s) | CHELA-SLAY (s) | Cumsum mem (MB) |
|---|---|---|---|
| 128 | 1.90 | 4.2 | 8192 |
| 256 | 3.50 | 7.5 | 16384 |
| 512 | 12.0 | 25.0 | 32768 |
| 1024 | 45.0 | 92.0 | 65536 |

## C.8 Gate Statistics

| Model | Mean g | Std g | Min g | Max g |
|---|---|---|---|---|
| DistilBERT | 0.92 | 0.05 | 0.75 | 0.99 |
| RoBERTa | 0.90 | 0.06 | 0.70 | 0.99 |
| GPT-2 | 0.91 | 0.05 | 0.72 | 0.99 |

## C.9 Training Loss

### DistilBERT
```
Step  Loss
1     0.35
5     0.20
10    0.15
20    0.10
30    0.07
40    0.05
50    0.04
```

## C.10 Reproduction Commands

```bash
python run_all_4_chela_slay.py --models distilbert,roberta,gpt2
```

---