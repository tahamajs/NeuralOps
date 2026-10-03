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
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal |

Base weights from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `rank` (unused for EAT) | 32 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `prune_layer` | 0.85 |

**Note.** The `rank` parameter is inherited from `BaseAttention` but is not 
used to compress EAT's projections. EAT uses LoRA, not low-rank factorization.

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

Only LoRA parameters are trainable; base projections are frozen.

## 6.5 Evaluation

- **Quality:** Accuracy, F1 (binary, macro), precision, recall
- **Efficiency:** Total params, trainable params, peak VRAM, inference time
- **Injection:** Compressed layers count, fallback count

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
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_eat.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 min.

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `eat` | Our method |
| `eat_no_prune` | EAT without pruning |
| Prior modules | `nha`, `a3`, `aasvd` |

---