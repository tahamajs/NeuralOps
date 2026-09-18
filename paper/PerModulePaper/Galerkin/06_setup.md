<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention | Path |
|---|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Flat | Galerkin |
| RoBERTa | 12 | 768 | 12 | 124.6M | Nested | Galerkin |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal | Fallback |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 Galerkin Hyperparameters

| Hyperparameter | Value | Note |
|---|---|---|
| `rank` (basis dim $r$) | 64 | Default 64 |
| `basis_type` | `ortho_dct` | DCT-II |
| `layer_id` | 0..N_l | For logging |
| `dropout` | 0.1 | |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Recovery steps | 50 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

- **Accuracy, F1, Precision, Recall**
- **Params before/after**
- **Peak VRAM**
- **Inference time**
- **Reconstruction error** (per layer)

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
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --recovery_steps 50 \
    --lora_rank 16
```

Expected runtime: ~12 minutes.

## 6.9 Seeds

Single seed (42). No multi-seed runs.

## 6.10 Baselines

- `full` — unmodified model
- `galerkin` — our method
- Prior modules: `nha`, `nasc`, `a3`, etc.

---
