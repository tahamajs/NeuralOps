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

Majority baseline: 50.92%.

## 6.3 SVD-Linear Hyperparameters

| Hyperparameter | Value |
|---|---|
| `mode` | {{MODE}} (exact or linear) |
| `rank` (SVD) | {{RANK}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `use_decay_gating` | {{USE_DECAY}} |
| `decay_activation` | softplus |
| `decay_channel_wise` | {{CHANNEL_WISE}} |
| `chunk_size` | 64 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1, precision, recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Parameter count vs baseline
- **Decay:** Final $\lambda$ distribution

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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_svd_linear_attention.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --mode {{MODE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `svd_linear` | Our method |
| Companion | `asvd`, `aasvd`, `rpc`, `svd_mamba` |
```

---