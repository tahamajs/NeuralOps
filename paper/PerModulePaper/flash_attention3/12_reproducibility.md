# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Optional: FlashAttention-3 (Hopper only)
pip install flash-attn --no-build-isolation
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |
| flash-attn | 1.0.0 (optional) |

## 12.3 Running

```bash
python run_all_4_flash3.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,flash3 \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_flash3.py
```

### Table 7.3 (Path used)
```bash
python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
python -c "from flash_attn import flash_attn_func; print('FA2 OK')"
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

| Model | Acc | Path |
|---|---|---|
| DistilBERT | 90.25% | models/weights.pt |
| RoBERTa | 78.2% | models/weights.pt |
| GPT-2 | 77.5% | 0.85 |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-flash3`.
- **Metrics:** `results/unified_flash3/*/summary.csv`.

---