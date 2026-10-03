# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_jamba_info.py \
    --models distilbert,roberta,gpt2 \
    --bottleneck_dim 64 \
    --recovery_steps 50
```

## 12.3 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_jamba_info.py
```

### Table 7.3 (Bottleneck Dim)
```bash
for b in 64 128 256 512; do
    python run_all_4_jamba_info.py --bottleneck_dim $b --output_suffix b$b
done
```

### Table 7.4 (No Bottleneck)
```bash
python run_all_4_jamba_info.py --bottleneck_dim 768  # full dim
```

## 12.4 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.5 Expected Results

| Model | Acc | Params Δ |
|---|---|---|
| DistilBERT | 89.68% | +48.5\%% |
| RoBERTa | 52.98% | +48.5\%% |
| GPT-2 | 50.80% | +48.5\%% |

## 12.6 Data Availability

- **Dataset:** SST-2.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-jamba_info`.
- **Metrics:** `results/unified_jamba_info/*/summary.csv`.

---