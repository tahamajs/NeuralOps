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

| Parameter | Value |
|---|---|
| `use_full_attention` | 0.85 |
| `d_model` | Auto from config |
| `num_heads` | Auto from config |
| `rank` (GatedDeltaNet) | 32 |

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

**Note.** The LayerNorm `self.norm` is **not** in the LoRA parameter set. 
It is trained directly (all parameters have `requires_grad=True`).

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params, Trainable params
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
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
# Path A (full attention)
python run_all_4_gated_deltanet_block.py \
    --models distilbert,roberta,gpt2 \
    --use_full_attention true

# Path B (GatedDeltaNet)
python run_all_4_gated_deltanet_block.py \
    --models distilbert,roberta,gpt2 \
    --use_full_attention false
```

## 6.9 Seeds

Single seed (42).

## 6.10 Diagnostics

To verify weight transfer, add logging:

```python
# In copy_weights:
logger.info(f"Q/K/V concat shape: {qkv_w.shape}")
logger.info(f"Target in_proj_weight shape: {self.attn.in_proj_weight.shape}")
logger.info(f"Match: {qkv_w.shape == self.attn.in_proj_weight.shape}")
```

## 6.11 Baselines

- `full` — unmodified model
- `gated_deltanet` — attention-level GatedDeltaNet
- `gated_deltanet_block` — this module

---