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

## 6.3 Mamba-Inspired SSM Hyperparameters

| Hyperparameter | Default |
|---|---|
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `A_log` init | $\log(1), \log(2), \ldots, \log(d)$ |
| `pres_gate` init | 2.0 |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| WD | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters
- Peak VRAM
- Inference time

### Complexity
- Memory scaling with $L$
- Time scaling with $L$

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
python run_all_4_mamba.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `mamba` | Our method |
| `mamba_no_pres` | Without preservation gate (ablation) |
| `mamba_no_decay` | Without learned decay (ablation) |
| `mamba_linear_decay` | Fixed linear decay (ablation) |

## 6.10 Seeds

Single seed ({{SEED}}). Limitation stated in Section 10.
```

---