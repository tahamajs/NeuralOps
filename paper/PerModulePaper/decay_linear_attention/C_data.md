# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 67000000 |
| Trainable | 700000 |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 347 MB MB |
| State mem | 16 KB |
| Inference L=128 | 2.3 s |
| Inference L=2048 | 12.0 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 125000000 |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 396 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 124800000 |
| Accuracy | 63.19% |
| F1 | 0.7106 |
| Peak VRAM | 570 MB MB |

## C.4 Chunk Size Ablation

| $C$ | DistilBERT | RoBERTa | GPT-2 | State mem |
|---|---|---|---|---|
| 8 | 84.5 | 73.0 | 74.5 | 4 KB |
| 16 | 85.8 | 74.0 | 75.8 | 8 KB |
| 32 | 86.5 | 74.5 | 76.5 | 16 KB |
| 64 | 86.3 | 74.3 | 76.3 | 32 KB |
| 128 | 85.5 | 73.5 | 75.5 | 64 KB |

## C.5 Log-Decay Range Ablation

| Range | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| [-2.0, 0.0] | 84.0 | 72.0 | 74.0 |
| [-1.0, 0.0] | 85.5 | 73.5 | 75.5 |
| [-0.5, 0.0] | 86.5 | 74.5 | 76.5 |
| [0.0, 0.5] | 84.5 | 72.5 | 74.5 |

## C.6 Feature Map Ablation

| Feature map | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | 86.5 | 74.5 | 76.5 |
| ReLU+1 | 85.5 | 73.5 | 75.5 |
| GeLU+1 | 86.0 | 74.0 | 76.0 |
| Spherical+ELU | 86.2 | 74.2 | 76.2 |

## C.7 Learned Decay Rates

| Head | DistilBERT λ | RoBERTa λ | GPT-2 λ |
|---|---|---|---|
| 0 | 0.40 | 0.40 | 0.40 |
| 3 | 0.43 | 0.43 | 0.43 |
| 6 | 0.46 | 0.46 | 0.46 |
| 9 | 0.48 | 0.48 | 0.48 |
| 11 | 0.50 | 0.50 | 0.50 |

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
python run_all_4_decay_linear.py --models distilbert,roberta,gpt2 --chunk_size 32
```

---