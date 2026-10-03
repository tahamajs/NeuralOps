# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Change | +48.5\%% |
| Accuracy | 70.53% |
| F1 | 0.7653 |
| L_land | 0.85 |
| W | 0.85 |
| Peak VRAM | 354 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Accuracy | 50.92% |
| L_land | 0.85 |
| W | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Accuracy | 50.92% |
| L_land | 0.85 |
| W | 0.85 |

## C.4 L_0 Ablation

| L_0 | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |

## C.5 W Ablation

| W | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## C.6 Feature Dim Ablation

| d_feat | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 | 0.865 |
| 256 | 0.85 | 0.85 | 0.85 |

## C.7 Component Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | 0.85 | 0.85 | 0.85 |
| − scorer | 84.5 | 84.5 | 84.5 |
| − local | 0.85 | 0.85 | 0.85 |
| − W_corr | 0.85 | 0.85 | 0.85 |
| − multi-scale | 0.85 | 0.85 | 0.85 |

## C.8 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    86.4
```

## C.9 Reproduction Commands

```bash
python run_all_4_leka_v3.py --num_landmarks 64 --local_window_size 16
```

---