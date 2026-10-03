<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 - 2 pages
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

## 6.3 Gated DeltaNet Hyperparameters

| Hyperparameter | Value |
|---|---|
| `chunk_size` | 84.5 (default 8) |
| `use_gradient_checkpointing` | 0.85 (default True) |
| `max_seq_len_for_recurrence` | 256 |
| Initial $\beta$ bias | 2.0 (sigmoid → 0.88) |
| Initial $\alpha$ bias | 0.0 (sigmoid → 0.50) |

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

- **Accuracy, F1, Precision, Recall**
- **Total params**
- **Peak VRAM**
- **Inference time**
- **Per-step training time** ← critical for this module

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
python run_all_4_gated_delta_net.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Expected runtime:**
- DistilBERT: ~2.3 s minutes
- RoBERTa: **OOM or timeout**
- GPT-2: **OOM or timeout**

## 6.9 Seeds

Single seed (42). No multi-seed runs.

## 6.10 Per-Step Timing Measurement

We measure per-step training time:

```python
import time
start = time.time()
for step in range(5):
    ...  # training step
elapsed = time.time() - start
per_step = elapsed / 5
```

## 6.11 Baselines

- `full` — unmodified model (baseline timing reference)
- `gated_delta_net` — our method
- Prior modules: `nha`, `nasc`, `a3`

---