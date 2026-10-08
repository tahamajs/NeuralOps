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
| `num_clusters` | 16 |
| `n_iters` | 3 |
| `window_size` | 32 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| Temperature init | 0.1 |
| Blend init | 0.5 |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
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

RTX 4070 Laptop, CUDA 12.1, PyTorch 2.4.0.

## 6.8 Reproduction

```bash
python run_all_4_cluster.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters 16 \
    --window_size 32 \
    --recovery_steps 50
```

---