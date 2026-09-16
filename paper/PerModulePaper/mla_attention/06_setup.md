<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128 tokens.

## 6.3 MLA Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Latent dimension $c$ | {{RANK}} | MLA rank |
| `lora_rank` | {{LORA_RANK}} | Q, O adapters |
| `lora_alpha` | {{LORA_ALPHA}} | Scaling |
| `output_scale` init | 1.0 | Trainable |
| `dropout` | 0.1 | Standard |

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
- Accuracy, F1, Precision, Recall

### Efficiency
- Total parameters
- **KV-cache size per token** (unique to MLA)
- Peak inference VRAM
- Inference time

### Compression
- Latent dimension $c$
- Effective K/V reconstruction error
- Normalization scale $\gamma$

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
| Transformers | {{HF_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_mla.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} minutes.

## 6.9 Seeds

All experiments use seed {{SEED}}.

## 6.10 Baselines

1. **`full`:** Uncompressed model.
2. **`mla`:** Our reproduction.
3. **Companion methods:** `minimal_svd`, `minimal_prism`, `svd_llm_v2`.
```

---