# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV Heads | Params |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

Base weights from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `k_rank` | {{RANK}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `dropout` | 0.1 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| WD | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters (baseline, ours).
- Attention parameters.
- Peak VRAM.
- Inference time.

### Compression
- Reconstruction error per projection.
- Actual compression ratio.

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_soa_fused.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `soa_fused` | Our method |
| `soa_fused_k64` | Rank 64 (ablation) |
| `soa_fused_k192` | Rank 192 (ablation) |
| Vanilla SVD | Unweighted SVD (same as ours, but without LoRA) |

## 6.10 Seeds

Single seed ({{SEED}}).

---