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
| `n_experts` | 8 |
| `capacity_factor` | 1.25 |
| `noise_std` | 0.05 |
| `moe_init_gate` | 0.1 |

## 6.4 LoRA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |

## 6.5 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch | 32 |
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
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.9 Reproduction

```bash
python run_all_4_expert_choice_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts 8 \
    --capacity_factor 1.25 \
    --recovery_steps 50
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

Single seed (42).

---