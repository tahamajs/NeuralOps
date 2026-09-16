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
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013).

Sequence length truncated to 128.

## 6.3 Performer Configuration

| Parameter | Value |
|---|---|
| `kernel_type` | {{KERNEL_TYPE}} (`favor_plus`) |
| `num_features` | {{NUM_FEATURES}} (default 64) |
| `eps` | $10^{-6}$ |
| `dropout` | 0.1 |

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
- Total params
- Peak VRAM
- Inference time

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
python run_all_4_performer.py \
    --models distilbert,roberta,gpt2 \
    --num_features {{NUM_FEATURES}} \
    --kernel_type {{KERNEL_TYPE}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Seeds

Single seed ({{SEED}}).

## 6.10 Ablations

- `num_features`: 32, 64, 128, 256
- `kernel_type`: `favor_plus`, `elu`, `relu`
- LoRA rank: 4, 8, 16, 32

## 6.11 Baselines

- `full` — unmodified model
- `performer` — our method
- Prior modules: `cosformer`, `orion`, `a3`
```

---