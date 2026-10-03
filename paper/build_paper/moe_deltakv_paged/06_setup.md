# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params | Causal |
|---|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M | No |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M | Yes |

## 6.2 Dataset

**SST-2**:

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%.

## 6.3 MoE-DeltaKV-Paged Hyperparameters

| Hyperparameter | Default | Notes |
|---|---|---|
| `page_size` | 64 | tokens per page |
| `k_experts` | 4 | top-k pages per query |
| `svd_rank` | 16 | delta compression rank |
| `lora_rank` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling |
| `W_rg` init | identity | router weight |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch size | 32 |
| Precision | FP16 |
| Aux loss weight | 0.01 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters
- Trainable parameters
- Peak VRAM
- Inference time
- Number of selected pages (mean)

### Router Analysis
- Page selection distribution
- Aux loss value
- Page-0 selection rate

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
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_moe_deltakv_paged.py \
    --models distilbert,roberta,gpt2 \
    --page_size 64 \
    --k_experts 4 \
    --svd_rank 16 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `moe_deltakv_paged` | Our method |
| `moba_only` | MoBA without Delta-KV |
| `delta_kv_only` | Delta-KV without MoBA |
| `no_aux_loss` | Without load balancing |

## 6.10 Seeds

Single seed (42). Limitation stated in Section 10.

---