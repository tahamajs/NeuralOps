<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 SALT Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Rank $r$ | {{RANK}} | SVD and residual |
| $\gamma$ init | 1.0 | — |
| $A$ init | $\mathcal{N}(0, 0.01^2)$ | — |
| $B$ init | 0 | — |
| Freeze SVD | Yes | Buffers |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

**Note.** SALT has no separate LoRA adapters — the trainable parameters are 
$\gamma, A, B$.

## 6.5 Evaluation

### 6.5.1 Quality Metrics

- Accuracy, F1, Precision, Recall

### 6.5.2 Efficiency Metrics

- Total params
- Trainable params
- Storage size (MB on disk)
- Peak VRAM at inference
- Inference time

**Storage size is the primary SALT benefit.** Inference VRAM is not reduced 
(naive forward).

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop (8 GB) |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_salt.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Seeds

Single seed {{SEED}}.

## 6.10 Baselines

- `full` — unmodified model
- `salt` — our method
- Prior modules: `nha`, `a3`, `rala`, `routing`

---