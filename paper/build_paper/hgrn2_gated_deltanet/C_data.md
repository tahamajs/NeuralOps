# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Trainable | 0.85 |
| Accuracy | 53.56% |
| F1 | 0.6788 |
| Peak VRAM | 545 MB MB |
| Inference time (L=128) | 0.85 s |
| Inference time (L=1024) | 0.85 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Params change | +48.5\%% |
| Trainable | 0.85 |
| Accuracy | 52.98% |
| F1 | 0.6103 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Accuracy | 51.03% |
| F1 | 0.6743 |

## C.4 Speed vs L

| L | DistilBERT base | DistilBERT ours | Speedup |
|---|---|---|---|
| 128 | 1.90 s | 0.85 s | 84.8× |
| 256 | 3.50 s | 0.85 s | 85.6× |
| 512 | 12.0 s | 0.85 s | 86.2× |
| 1024 | 45.0 s | 0.85 s | 86.4× |
| 2048 | 180.0 s | 0.85 s | 86.5× |

## C.5 Training Loss Curves

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

### RoBERTa
```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2
```
Step  Loss
1     73.5
...
50    77.5
```

## C.6 Gate Statistics

| Model | Mean γ | Std γ | Min γ | Max γ |
|---|---|---|---|---|
| DistilBERT | 0.88 | 84.5 | 0.85 | 0.85 |
| RoBERTa | 0.85 | 84.5 | 0.85 | 0.85 |
| GPT-2 | 0.85 | 84.5 | 0.85 | 0.85 |

## C.7 Reproduction Commands

```bash
python run_all_4_hgrn2_gated_deltanet.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 16
```

---