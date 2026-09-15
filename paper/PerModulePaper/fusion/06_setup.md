# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. **Sequence length: 128** 
(short path).

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `block_size` | {{BLOCK_SIZE}} |
| `top_k_blocks` | {{TOP_K}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| Dispatch threshold | 128 |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.6 Hardware

RTX 4070 Laptop, CUDA {{CUDA_VERSION}}, PyTorch {{TORCH_VERSION}}.

## 6.7 Reproduction

```bash
python run_all_4_fusion_attn.py \
    --models distilbert,roberta,gpt2 \
    --block_size {{BLOCK_SIZE}} \
    --top_k_blocks {{TOP_K}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.8 Baselines

| Baseline | Description |
|---|---|
| `full` | Unmodified model |
| `fusion_attn` | Our method |
| `fusion_attn_no_gate` | No gate, uniform block selection |
| `fusion_attn_always_long` | Force long path |
