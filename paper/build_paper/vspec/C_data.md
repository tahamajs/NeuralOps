# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 68000000 |
| Attention params before | 2,359,296 |
| Attention params after | 1070000 |
| Attention ratio | 2.2× |
| Accuracy | 87.5% |
| F1 | 0.875 |
| Peak VRAM | 195 MB |
| Inference time | 2.2 s |
| Mean spectral gate $g$ | 0.92 |
| Mean head gate $h$ | 0.85 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 126000000 |
| Attention params | 1070000 |
| Accuracy | 75.0% |
| F1 | 0.750 |
| Mean $g$ | 0.90 |
| Mean $h$ | 0.82 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 126000000 |
| Attention params | 1070000 |
| Accuracy | 77.0% |
| F1 | 0.770 |
| Mean $g$ | 0.91 |
| Mean $h$ | 0.84 |

## C.4 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| 32 | 83.0 | 4.5× | 70.0 | 4.5× |
| 64 | 86.0 | 3.0× | 73.0 | 3.0× |
| 128 | 87.5 | 2.2× | 75.0 | 2.2× |
| 192 | 87.8 | 1.7× | 75.2 | 1.7× |
| 256 | 87.9 | 1.4× | 75.3 | 1.4× |

## C.5 Gate Ablations

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full VSpec++ | 87.5 | 75.0 | 77.0 |
| No router | 88.0 | 75.5 | 77.2 |
| No spectral gate | 85.0 | 72.5 | 74.5 |
| No head gate | 86.5 | 74.2 | 75.8 |
| Deterministic | 87.2 | 74.7 | 76.7 |

## C.6 Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 0.15 | 70.0 | 72.0 |
| 25 | 85.5 | 73.5 | 75.5 |
| 50 | 0.05 | 75.0 | 77.0 |
| 100 | 87.7 | 75.2 | 77.2 |

## C.7 Per-Layer Gates (DistilBERT)

| Layer | $g_q$ | $g_k$ | $g_v$ | $h$ |
|---|---|---|---|---|
| 0 | 0.94 | 0.93 | 0.95 | 0.88 |
| 1 | 0.92 | 0.91 | 0.93 | 0.86 |
| 2 | 0.91 | 0.90 | 0.92 | 0.85 |
| 3 | 0.89 | 0.88 | 0.91 | 0.83 |
| 4 | 0.88 | 0.87 | 0.90 | 0.82 |
| 5 | 0.86 | 0.85 | 0.89 | 0.81 |

## C.8 Training Loss

### DistilBERT
```
Step  Loss
1     0.35
5     0.22
10    0.15
20    0.10
30    0.08
40    0.06
50    0.05
```

## C.9 Reproduction Commands

```bash
python run_all_4_vspec.py --models distilbert,roberta,gpt2
```

---