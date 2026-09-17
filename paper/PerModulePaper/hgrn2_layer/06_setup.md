# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

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
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `res_clip` | 30.0 |
| `chunk_size` | 32 |
| `freeze_base` | True |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch | 32 |
| Precision | FP16 |

Trainable: gate networks + LoRA (~2.5M per layer).

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Trainable parameters.
- Peak VRAM.
- Inference time at L = 128, 256, 512, 1024, 2048.

### Recurrence-specific
- Chunk size effect.
- Bidirectional vs causal.

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
python run_all_4_hgrn2.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `hgrn2` | Our method |
| `hgrn2_no_bidi` | Causal only (ablation) |
| `hgrn2_small_chunk` | Chunk size 16 (ablation) |

## 6.10 Seeds

Single seed (42).

---