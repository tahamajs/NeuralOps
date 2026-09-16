# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Params change | +{{D_PARAM_PCT}}% |
| Trainable | {{D_TRAINABLE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time (L=128) | {{D_T128}} s |
| Inference time (L=1024) | {{D_T1024}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Params change | +{{R_PARAM_PCT}}% |
| Trainable | {{R_TRAINABLE}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |

## C.4 Speed vs L

| L | DistilBERT base | DistilBERT ours | Speedup |
|---|---|---|---|
| 128 | 1.90 s | {{D_T128}} s | {{S128}}× |
| 256 | 3.50 s | {{D_T256}} s | {{S256}}× |
| 512 | 12.0 s | {{D_T512}} s | {{S512}}× |
| 1024 | 45.0 s | {{D_T1024}} s | {{S1024}}× |
| 2048 | 180.0 s | {{D_T2048}} s | {{S2048}}× |

## C.5 Training Loss Curves

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

### RoBERTa
```
Step  Loss
1     {{R_S1}}
...
50    {{R_S50}}
```

### GPT-2
```
Step  Loss
1     {{G_S1}}
...
50    {{G_S50}}
```

## C.6 Gate Statistics

| Model | Mean γ | Std γ | Min γ | Max γ |
|---|---|---|---|---|
| DistilBERT | {{D_GATE_MEAN}} | {{D_GATE_STD}} | {{D_GATE_MIN}} | {{D_GATE_MAX}} |
| RoBERTa | {{R_GATE_MEAN}} | {{R_GATE_STD}} | {{R_GATE_MIN}} | {{R_GATE_MAX}} |
| GPT-2 | {{G_GATE_MEAN}} | {{G_GATE_STD}} | {{G_GATE_MIN}} | {{G_GATE_MAX}} |

## C.7 Reproduction Commands

```bash
python run_all_4_hgrn2_gated_deltanet.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 16
```

---