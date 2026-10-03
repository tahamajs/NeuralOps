# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |

## 12.3 Running

```bash
python run_all_4_expert_choice_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts 8 \
    --capacity_factor 1.25 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,expert_choice_moe \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_expert_choice_moe.py
```

### Table 7.5 (Expert Count)
```bash
for e in 4 8 16; do
    python run_all_4_expert_choice_moe.py --n_experts $e --output_suffix e$e
done
```

### Table 7.6 (Capacity Factor)
```bash
for cf in 0.5 1.0 1.5 2.0; do
    python run_all_4_expert_choice_moe.py --capacity_factor $cf --output_suffix cf$cf
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

| Model | Acc | Params Δ | Expert balance |
|---|---|---|---|
| DistilBERT | 90.14% | +48.5\%% | Perfect |
| RoBERTa | 84.29% | +48.5\%% | Perfect |
| GPT-2 | 80.50% | +48.5\%% | Perfect |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-expert_choice_moe`.
- **Metrics:** `results/unified_expert_choice_moe/*/summary.csv`.

---