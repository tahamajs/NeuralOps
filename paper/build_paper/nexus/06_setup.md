# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Causal |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | No |
| RoBERTa | 12 | 768 | 12 | 124.6M | No |
| GPT-2 | 12 | 768 | 12 | 124.4M | Yes |

## 6.2 Dataset

**SST-2**: 67,349 train / 872 val / 1,821 test. Majority baseline 50.92%.
Max sequence length: 128.

**Important.** At $N = 128$, the stripe mask ($N > 256$) is inactive.

## 6.3 NEXUS Hyperparameters

| Hyperparameter | Default |
|---|---|
| `block_size` | 16 |
| `top_k_blocks` | 4 |
| `xattn_stride` | 2 |
| `seer_gate_rank` | 16 |
| `seer_top_k_blocks` | 4 |
| `seer_temperature` | 1.0 |
| `stripe_width` | 4 |
| `rectify_interval` | 4 |
| `rff_features` | 128 |
| `softpick_temperature` | 1.0 |
| `sparse_min_seq_len` | 8 |
| `enable_spectraformer` | False |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `recovery_steps` | 50 |
| Batch size | 32 |
| LR | 1e-4 |
| WD | 0.01 |

## 6.5 Evaluation

- Accuracy, F1
- Peak VRAM, inference time
- Active path (dense vs sparse)
- Router statistics

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW ≥ 50.92%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, 8 GB.

## 6.8 Reproduction

```bash
python run_all_4_nexus.py --models distilbert,roberta,gpt2 --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `nexus` | Full NEXUS |
| `nexus_xattn_only` | Only XAttention |
| `nexus_no_seer` | Without Seer gating |
| `nexus_no_stripe` | Without stripe |
| `nexus_no_softpick` | Standard softmax |
| `nexus_dense` | Force dense path |

---