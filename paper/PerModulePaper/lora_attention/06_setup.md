# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 LoRA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `output_scale` init | 1.0 |
| Base weights | Frozen |
| Dropout | 0.1 |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Trainable parameters
- Training memory
- Training time
- Inference time (before/after merge)

### Comparison
- Full fine-tuning (baseline)
- LoRA on FFN only (ablation)
- LoRA on Q/V only (ablation)
- LoRA on all four projections (ours)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_lora_attention.py \
    --models distilbert,roberta,gpt2 \
    --lora_rank {{LORA_RANK}} \
    --lora_alpha {{LORA_ALPHA}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention, no LoRA |
| `lora_attention` | Our method (LoRA on all four) |
| `lora_qv` | LoRA only on Q, V |
| `lora_ffn` | LoRA only on FFN |
| `full_ft` | Full fine-tuning (compute-expensive) |

## 6.10 Seeds

Single seed ({{SEED}}). Limitation stated in Section 10.

---