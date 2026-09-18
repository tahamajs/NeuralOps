<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Default path |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Dense |
| RoBERTa | 12 | 768 | 12 | 124.6M | Dense |
| GPT-2 | 12 | 768 | 12 | 124.4M | Cascade |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 NASC Configuration

| Parameter | Value |
|---|---|
| `window_size` | 128 (default 64) |
| `block_size` | 64 (default 32) |
| `top_k` | 2 (default 2) |
| `rff_features` | 64 (default 64) |
| `enable_cascade_on_encoders` | 0.85 |

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
- **Gate distribution** (mean $g_0, g_1, g_2$)

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
python run_all_4_nasc.py \
    --models distilbert,roberta,gpt2 \
    --window_size 128 \
    --block_size 64 \
    --top_k 2 \
    --rff_features 64 \
    --recovery_steps 50
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- `enable_cascade_on_encoders`: True/False
- `window_size`: 16, 32, 64, 128
- `block_size`: 16, 32, 64
- `top_k`: 1, 2, 3, 4
- `rff_features`: 32, 64, 128

## 6.11 Baselines

- `full` — unmodified model
- `nasc` — our method
- Prior modules: `cosformer`, `fasa`, `a3`

---