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

## 6.3 Hyperparameters

### Linformer-specific

| Hyperparameter | Value |
|---|---|
| `linformer_k` | {{K}} |
| `max_seq_len` | {{MAX_SEQ}} |
| E, F init | diagonal + noise 0.02 |

### LoRA Recovery

| Hyperparameter | Value |
|---|---|
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `recovery_steps` | {{RECOVERY_STEPS}} |
| `batch_size` | {{BATCH_SIZE}} |
| LR | {{LR}} |
| WD | {{WD}} |

## 6.4 Evaluation

### Quality
- Accuracy, F1, precision, recall

### Efficiency
- Params (before/after)
- Peak VRAM
- Inference time
- Effective complexity ($L \cdot k$)

## 6.5 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.6 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.7 Reproduction

```bash
python run_all_4_linformer.py \
    --models distilbert,roberta,gpt2 \
    --linformer_k {{K}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.8 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `linformer` | Our implementation |
| `linformer_no_E` | E = identity (ablation) |
| `linformer_random_E` | Random E init (ablation) |
| `linformer_no_causal_fallback` | Break causality (ablation) |

## 6.9 Seeds

Single seed ({{SEED}}). Limitation stated in Section 10.

---