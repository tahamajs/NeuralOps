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
| `rank` (basis dim $r$) | {{RANK}} | Default 64 |
| `basis_type` | `ortho_dct` | DCT-II |
| `layer_id` | 0..N_l | For logging |
| `dropout` | 0.1 | |

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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Expected runtime: ~{{RUNTIME}} minutes.

## 6.9 Seeds

Single seed ({{SEED}}). No multi-seed runs.

## 6.10 Baselines

- `full` — unmodified model
- `galerkin` — our method
- Prior modules: `nha`, `nasc`, `a3`, etc.
```

---