<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%.

## 6.3 OHA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `window_size` $W$ | 64 |
| `compression_rank` $d_c$ | 32 |
| `gate_bias` | 2.0 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `rank` | 64 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Not applicable (OHA adds parameters)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_oha.py \
    --models distilbert,roberta,gpt2 \
    --window_size 64 \
    --compression_rank 32 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `oha` | Our method |
| Companion | `nha`, `eat`, `infini` |

---
