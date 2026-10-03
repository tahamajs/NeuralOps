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

## 6.3 ASVD Hyperparameters

| Hyperparameter | Value |
|---|---|
| `rank` | 64 |
| `dynamic_rank` | true |
| `energy_threshold` $\varepsilon$ | 0.95 |
| `error_threshold` $\tau$ | 0.20 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |

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

- **Quality:** Accuracy, F1, Precision, Recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Compression ratio, per-layer rank distribution, reconstruction error

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
python run_all_4_asvd.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --energy_threshold 0.95 \
    --error_threshold 0.20 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `asvd` | Our method |
| Companion | `a3`, `aasvd`, `sdrm` |

---