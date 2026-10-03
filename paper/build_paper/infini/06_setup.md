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

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Infini-Attention Hyperparameters

| Hyperparameter | Value |
|---|---|
| `compress_dim` $d_c$ | 64 |
| `local_window` $W$ | 128 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `rank` | 32 |
| $\beta$ init | -5.0 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1, Precision, Recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Memory usage:** Per-layer memory matrix size

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
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_infini.py \
    --models distilbert,roberta,gpt2 \
    --compress_dim 64 \
    --local_window 128 \
    --recovery_steps 50
```

Runtime: ~PyTorch 2.0 min.

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `infini` | Our method |
| Companion methods | `nha`, `eat`, `aasvd` |

---