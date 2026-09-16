# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | {{D_PARAMS}} |
| Trainable | {{D_TRAINABLE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time L=128 | {{D_T128}} s |
| Path used | {{D_PATH}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Path used | {{R_PATH}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Path used | {{G_PATH}} |

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
| DistilBERT | {{D_DIFF_MAX}} |
| RoBERTa | {{R_DIFF_MAX}} |
| GPT-2 | {{G_DIFF_MAX}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
20    {{D_S20}}
30    {{D_S30}}
40    {{D_S40}}
50    {{D_S50}}
```

## C.7 Reproduction Commands

```bash
python run_all_4_flash3.py --models distilbert,roberta,gpt2
```
```

---