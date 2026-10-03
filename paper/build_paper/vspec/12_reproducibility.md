# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_vspec.py \
    --models distilbert,roberta,gpt2 \
    --rank 128 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,vspec \
    --rank 128 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_vspec.py
```

### Table 7.4-7.6 (Ablations)
```bash
python run_all_4_vspec.py --no_router
python run_all_4_vspec.py --no_spectral_gate
python run_all_4_vspec.py --no_head_gate
python run_all_4_vspec.py --deterministic
```

### Table 7.7 (Rank)
```bash
for rank in 32 64 128 192 256; do
    python run_all_4_vspec.py --rank $rank --output_suffix r$rank
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Attention Ratio |
|---|---|---|
| DistilBERT | 87.5% | 2.2× |
| RoBERTa | 75.0% | 2.2× |
| GPT-2 | 77.0% | 2.2× |

## 12.7 Data Availability

- **Dataset:** SST-2.
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-vspec`.
- **Metrics:** `results/unified_vspec/*/summary.csv`.

---