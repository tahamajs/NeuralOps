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
| Inference time | {{D_TIME}} s |
| Cumsum memory | {{CMEM_128}} MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Feature Dim Ablation

| F | DistilBERT | RoBERTa | GPT-2 | Memory (MB) |
|---|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} | {{CMEM_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} | {{CMEM_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} | {{CMEM_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} | {{CMEM_F256}} |

## C.5 Gate Bias Ablation

| $b_g$ | σ(b_g) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.50 | {{D_GB0}} | {{R_GB0}} | {{G_GB0}} |
| 2.0 | 0.88 | {{D_GB2}} | {{R_GB2}} | {{G_GB2}} |
| 5.0 | 0.993 | {{D_GB5}} | {{R_GB5}} | {{G_GB5}} |
| 10.0 | ~1.0 | {{D_GB10}} | {{R_GB10}} | {{G_GB10}} |

## C.6 Stream Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dense only | {{D_DENSE_ONLY}} | {{R_DENSE_ONLY}} | {{G_DENSE_ONLY}} |
| Linear only | {{D_LINEAR_ONLY}} | {{R_LINEAR_ONLY}} | {{G_LINEAR_ONLY}} |
| Hybrid | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |

## C.7 Sequence-Length Scaling

| L | Baseline (s) | CHELA-SLAY (s) | Cumsum mem (MB) |
|---|---|---|---|
| 128 | 1.90 | {{D_T128}} | {{CMEM_128}} |
| 256 | 3.50 | {{D_T256}} | {{CMEM_256}} |
| 512 | 12.0 | {{D_T512}} | {{CMEM_512}} |
| 1024 | 45.0 | {{D_T1024}} | {{CMEM_1024}} |

## C.8 Gate Statistics

| Model | Mean g | Std g | Min g | Max g |
|---|---|---|---|---|
| DistilBERT | {{D_GATE_MEAN}} | {{D_GATE_STD}} | {{D_GATE_MIN}} | {{D_GATE_MAX}} |
| RoBERTa | {{R_GATE_MEAN}} | {{R_GATE_STD}} | {{R_GATE_MIN}} | {{R_GATE_MAX}} |
| GPT-2 | {{G_GATE_MEAN}} | {{G_GATE_STD}} | {{G_GATE_MIN}} | {{G_GATE_MAX}} |

## C.9 Training Loss

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

## C.10 Reproduction Commands

```bash
python run_all_4_chela_slay.py --models distilbert,roberta,gpt2
```
```

---