# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):
- Train: 67,349
- Validation: 872
- Test: 1,821
- Majority: 50.92%

## 6.3 Res-SVD Hyperparameters

| Hyperparameter | Value |
|---|---|
| `base_rank` $r_b$ | 32 |
| `residual_rank` $r_r$ | 16 |
| `scaling` $\alpha$ | 1.0 |
| `factorize_base` | True |
| `freeze_base` | True |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Steps | 50 |
| Batch size | 32 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Base + residual parameter ratio

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

RTX 4070 Laptop, 8 GB VRAM.

## 6.8 Reproduction

```bash
python run_all_4_res_svd.py \
    --models distilbert,roberta,gpt2 \
    --base_rank 32 \
    --residual_rank 16 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `res_svd` | Our method |
| Companion | `asvd`, `aasvd`, `a3`, `svd_linear` |

---