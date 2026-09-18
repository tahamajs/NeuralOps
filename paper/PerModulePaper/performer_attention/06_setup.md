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
| `kernel_type` | Gaussian (`favor_plus`) |
| `num_features` | 64 (default 64) |
| `eps` | $10^{-6}$ |
| `dropout` | 0.1 |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Batch size | 32 |
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
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_performer.py \
    --models distilbert,roberta,gpt2 \
    --num_features 64 \
    --kernel_type Gaussian \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- `num_features`: 32, 64, 128, 256
- `kernel_type`: `favor_plus`, `elu`, `relu`
- LoRA rank: 4, 8, 16, 32

## 6.11 Baselines

- `full` — unmodified model
- `performer` — our method
- Prior modules: `cosformer`, `orion`, `a3`

---