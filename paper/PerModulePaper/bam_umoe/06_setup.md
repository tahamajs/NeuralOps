# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. Majority baseline: 50.92%.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `num_experts` | {{NUM_EXPERTS}} |
| `top_k` | {{TOP_K}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| MoE scale $\alpha$ | 0.1 |
| Gate bias $b_g$ | -3.0 |
| Noise std (training) | 0.1 |
| Expert multiplier | 2 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Metrics

- Accuracy, F1, precision, recall
- Parameter count (before/after)
- Peak VRAM
- Inference time
- MoE load balance (expert utilization)

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, 8 GB VRAM, CUDA {{CUDA_VERSION}}, PyTorch {{TORCH_VERSION}}.

## 6.8 Reproduction

```bash
python run_all_4_bam_umoe.py \
    --models distilbert,roberta,gpt2 \
    --num_experts {{NUM_EXPERTS}} \
    --top_k {{TOP_K}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

---