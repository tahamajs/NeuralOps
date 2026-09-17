# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 66.4M |
| Trainable | 0.85 |
| Accuracy | 90.25% |
| F1 | 0.9065 |
| Peak VRAM | 1335 MB MB |
| Inference time L=128 | 0.85 s |
| Path used | models/weights.pt |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 124.6M |
| Accuracy | 78.2% |
| F1 | 0.865 |
| Peak VRAM | 380 MB MB |
| Path used | models/weights.pt |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 124.4M |
| Accuracy | 77.5% |
| F1 | 0.865 |
| Peak VRAM | 380 MB MB |
| Path used | 0.85 |

## C.4 Sequence-Length Scaling

| L | Baseline | FA3 | FA3 FP8 |
|---|---|---|---|
| 128 | 1× | 1× | 1× |
| 256 | 1× | 1.5× | 2× |
| 512 | 1× | 2.5× | 4× |
| 1024 | 1× | 3.5× | 5.5× |
| 2048 | 1× | 4× | 6× |

(Speedups relative to baseline; only meaningful on Hopper.)

## C.5 Output Correctness

| Model | Max abs diff (FA3 vs SDPA) |
|---|---|
| DistilBERT | 0.85 |
| RoBERTa | 0.85 |
| GPT-2 | 0.85 |

## C.6 Training Loss

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

## C.7 Reproduction Commands

```bash
python run_all_4_flash3.py --models distilbert,roberta,gpt2
```

---