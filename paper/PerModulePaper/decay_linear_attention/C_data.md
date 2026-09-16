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
| State mem | 16 KB |
| Inference L=128 | {{D_T128}} s |
| Inference L=2048 | {{D_T2048}} s |

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

## C.4 Chunk Size Ablation

| $C$ | DistilBERT | RoBERTa | GPT-2 | State mem |
|---|---|---|---|---|
| 8 | {{D_C8}} | {{R_C8}} | {{G_C8}} | 4 KB |
| 16 | {{D_C16}} | {{R_C16}} | {{G_C16}} | 8 KB |
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} | 16 KB |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} | 32 KB |
| 128 | {{D_C128}} | {{R_C128}} | {{G_C128}} | 64 KB |

## C.5 Log-Decay Range Ablation

| Range | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| [-2.0, 0.0] | {{D_RANGE1}} | {{R_RANGE1}} | {{G_RANGE1}} |
| [-1.0, 0.0] | {{D_RANGE2}} | {{R_RANGE2}} | {{G_RANGE2}} |
| [-0.5, 0.0] | {{D_RANGE3}} | {{R_RANGE3}} | {{G_RANGE3}} |
| [0.0, 0.5] | {{D_RANGE4}} | {{R_RANGE4}} | {{G_RANGE4}} |

## C.6 Feature Map Ablation

| Feature map | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| ReLU+1 | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |
| GeLU+1 | {{D_GELU}} | {{R_GELU}} | {{G_GELU}} |
| Spherical+ELU | {{D_SPH}} | {{R_SPH}} | {{G_SPH}} |

## C.7 Learned Decay Rates

| Head | DistilBERT λ | RoBERTa λ | GPT-2 λ |
|---|---|---|---|
| 0 | {{LAMBDA_D_H0}} | {{LAMBDA_R_H0}} | {{LAMBDA_G_H0}} |
| 3 | {{LAMBDA_D_H3}} | {{LAMBDA_R_H3}} | {{LAMBDA_G_H3}} |
| 6 | {{LAMBDA_D_H6}} | {{LAMBDA_R_H6}} | {{LAMBDA_G_H6}} |
| 9 | {{LAMBDA_D_H9}} | {{LAMBDA_R_H9}} | {{LAMBDA_G_H9}} |
| 11 | {{LAMBDA_D_H11}} | {{LAMBDA_R_H11}} | {{LAMBDA_G_H11}} |

## C.8 Training Loss

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

## C.9 Reproduction Commands

```bash
python run_all_4_decay_linear.py --models distilbert,roberta,gpt2 --chunk_size 32
```

---