# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

Base weights from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `rank` | 32 |
| `recovery_steps` | 50 |
| `batch_size` | 32 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 0.01 |
| `precision` | FP16 |

## 6.4 Recovery Fine-Tuning

Trainable params:
- LoRA on Q, K, V, O and branch projections.
- Gate.

Frozen:
- All base weights.

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters (baseline, ours).
- Trainable parameters.
- Peak VRAM.
- Inference time (median of 20 runs at L = 128, 256, 512, 1024).
- Training time per step.

### Recurrence-specific
- Speedup vs baseline at each L.
- Breakeven sequence length $L^*$.

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
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_hgrn2_gated_deltanet.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `hgrn2_gated_deltanet` | Our method |
| `hgrn2_only` | HGRN-2 branch alone (ablation) |
| `deltanet_only` | DeltaNet branch alone (ablation) |
| `uniform_gate` | Fixed gate γ = 0.5 (ablation) |

## 6.10 Seeds

Single seed (42). Multi-seed limitation stated in Section 10.

---