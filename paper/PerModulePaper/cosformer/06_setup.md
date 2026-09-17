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

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 Configuration

| Hyperparameter | Value |
|---|---|
| `act_fun` | `relu` (ELU+1) |
| `max_seq_len` | 512 |
| `eps` | $10^{-6}$ |
| `chunk_size` | 32 |
| `rank` (basis) | 64 |

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

- Accuracy, F1, Precision, Recall
- Total params
- Peak VRAM
- Inference time
- Per-step training time

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
python run_all_4_cosformer.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 16 \
    --batch_size 8
```

## 6.9 Seeds

Single seed (42). No multi-seed runs.

## 6.10 Baselines

- `full` — unmodified model
- `cosformer` — our method
- Prior modules: `nha`, `nasc`, `a3`, `gated_delta_net`

---