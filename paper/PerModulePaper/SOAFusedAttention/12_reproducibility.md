# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running

```bash
python run_all_4_soa_fused.py \
    --models distilbert,roberta,gpt2 \
    --rank 128 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,soa_fused \
    --rank 128 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_soa_fused.py
```

### Table 7.5 (Rank Ablation)
```bash
for k in 32 64 128 192 384; do
    python run_all_4_soa_fused.py --rank $k --output_suffix k$k
done
```

### Table 7.6 (LoRA Rank)
```bash
for r in 4 8 16 32; do
    python run_all_4_soa_fused.py --lora_rank $r --output_suffix lr$r
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc | Attention ratio |
|---|---|---|
| DistilBERT | 81.31% | 1.92× |
| RoBERTa | 50.92% | 1.92× |
| GPT-2 | 50.92% | 1.92× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-soa_fused`.
- **Metrics:** `results/unified_soa_fused/*/summary.csv`.
- **No calibration data required.**

---
