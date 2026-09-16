<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal Conv1D |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 (50.2%) | 33,542 |
| Validation | 872 | 444 (50.9%) | 428 |
| Test | 1,821 | — | — |

Majority baseline: **50.92%**. Sequence length: 128 tokens.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| $r_\text{max}$ | {{RANK}} | Rank cap |
| $\eta$ (energy) | {{ENERGY}} | Spectral energy threshold |
| `dropout` | 0.1 | Standard |
| `lora_rank` | {{LORA_RANK}} | Recovery adapter |
| `lora_alpha` | {{LORA_ALPHA}} | Scaling = alpha / rank |
| `target_attention_only` | True | Only attention projections |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality

- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency

- Total parameters (before / after)
- Compression ratio: baseline / compressed
- Peak inference VRAM (MB)
- Inference time (median of 20 runs, batch 8, seq 128)

### Compression

- Effective rank per layer
- Reconstruction error (Frobenius)
- Compressed layers (count)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% = majority baseline + 3·SE ($\text{SE} \approx 1.69\%$).

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_minimal_prism.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --energy {{ENERGY}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} minutes.

## 6.9 Seeds

All experiments use seed {{SEED}}. Single-seed due to compute budget.

## 6.10 Baselines

1. **`full`:** Uncompressed pretrained model.
2. **`minimal_prism`:** Our method.
3. **Ablations:** `minimal_prism_fixed_rank`, `minimal_prism_no_lora`.
4. **Companion methods:** `a3`, `aasvd`, `sdrm`, `minimal_nova`, `svd_linear`.

---