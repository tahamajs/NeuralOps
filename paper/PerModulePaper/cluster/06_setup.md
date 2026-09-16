# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. Majority baseline: 50.92%.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `num_clusters` | {{NUM_CLUSTERS}} |
| `n_iters` | {{N_ITERS}} |
| `window_size` | {{WINDOW_SIZE}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| Temperature init | 0.1 |
| Blend init | 0.5 |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Metrics

- Accuracy, F1
- Attention FLOPs (theoretical)
- Peak VRAM
- Inference time
- Cluster quality (silhouette-like metric)

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, CUDA {{CUDA_VERSION}}, PyTorch {{TORCH_VERSION}}.

## 6.8 Reproduction

```bash
python run_all_4_cluster.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters {{NUM_CLUSTERS}} \
    --window_size {{WINDOW_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```
```

---