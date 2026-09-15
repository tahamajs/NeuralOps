# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 MoE Hyperparameters

| Hyperparameter | Value |
|---|---|
| `n_experts` | {{N_EXPERTS}} |
| `capacity_factor` | {{CAPACITY_FACTOR}} |
| `noise_std` | 0.05 |
| `moe_init_gate` | {{MOE_INIT_GATE}} |

## 6.4 LoRA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |

## 6.5 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| WD | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.6 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Trainable parameters.
- Peak VRAM.
- Inference time.

### MoE-Specific
- Expert utilization (should be perfectly balanced).
- Gate evolution.
- Routing entropy.

## 6.7 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.8 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.9 Reproduction

```bash
python run_all_4_expert_choice_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts {{N_EXPERTS}} \
    --capacity_factor {{CAPACITY_FACTOR}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.10 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `expert_choice_moe` | Our method |
| `expert_choice_moe_top1` | capacity 0.5 (ablation) |
| `expert_choice_moe_top2` | capacity 2.0 (ablation) |
| BAM-UMoE | Token-choice MoE |

## 6.11 Seeds

Single seed ({{SEED}}).

---
