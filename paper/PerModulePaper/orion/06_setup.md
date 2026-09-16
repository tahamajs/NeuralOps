<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | SVD rank |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | 64 |
| RoBERTa | 12 | 768 | 12 | 124.6M | 64 |
| GPT-2 | 12 | 768 | 12 | 124.4M | 64 |

## 6.2 Dataset

**SST-2** (Socher et al., 2013).

## 6.3 ORION Configuration

| Parameter | Value |
|---|---|
| `svd_energy` | {{SVD_ENERGY}} (default 0.95) |
| `svd_min_rank` | {{SVD_MIN_RANK}} (default 4) |
| `svd_max_rank` | {{SVD_MAX_RANK}} (default 64) |
| `num_kernel_heads` | Auto: 2/3 of heads |
| `num_full_heads` | Auto: 1/3 of heads |
| `rff_features` | {{RFF_FEATURES}} (default 256) |
| `softpick_temperature` | {{TEMPERATURE}} (default 1.0) |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| LoRA rank | {{LORA_RANK}} |
| LoRA alpha | {{LORA_ALPHA}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params (before/after)
- Peak VRAM
- Inference time
- Parameter reduction ratio

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_orion.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy {{SVD_ENERGY}} \
    --svd_max_rank {{SVD_MAX_RANK}} \
    --rff_features {{RFF_FEATURES}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Seeds

Single seed ({{SEED}}).

## 6.10 Ablations

- SVD energy: 0.90, 0.95, 0.99
- SVD max rank: 32, 64, 128
- RFF features: 128, 256, 512
- Head split: 2/3:1/3, 1/2:1/2, 3/4:1/4

## 6.11 Baselines

- `full` — unmodified model
- `orion` — our method
- Prior modules: `a3`, `cosformer`, `nasc`
```

---