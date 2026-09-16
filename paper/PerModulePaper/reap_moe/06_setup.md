# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

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
| `n_experts` $E$ | {{N_EXPERTS}} |
| `top_k` $k$ | {{TOP_K}} |
| `noise_std` $\sigma$ | 0.05 |
| `use_svd` | True |
| SVD rank $r$ | $d/4$ |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| LoRA rank | {{LORA_RANK}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall
- Validation loss

### Efficiency
- Total parameters
- Expert utilization (from aux loss)
- Peak VRAM
- Inference time

### MoE-specific
- Router entropy
- Load balance metric (max/mean expert usage)
- Auxiliary loss value

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

- GPU: NVIDIA RTX 4070 Laptop (8 GB)
- CUDA: {{CUDA_VERSION}}
- PyTorch: {{TORCH_VERSION}}
- Transformers: {{HF_VERSION}}

## 6.8 Reproduction

```bash
python run_all_4_reap_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts {{N_EXPERTS}} \
    --top_k {{TOP_K}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed |
| `reap_moe` | Our method |
| `reap_moe_no_svd` | Dense experts |
| `reap_moe_no_aux` | No load-balancing loss |
| `reap_moe_no_noise` | Deterministic routing |

## 6.10 Seeds

All experiments use seed {{SEED}}.

---