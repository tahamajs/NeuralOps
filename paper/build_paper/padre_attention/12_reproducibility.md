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
python run_all_4_padre.py \
    --models distilbert,roberta,gpt2 \
    --order 2 \
    --recovery_steps 50
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,padre \
    --order 2
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_padre.py
```

### Table 7.2 (Order)
```bash
for d in 1 2 4 8; do
    python run_all_4_padre.py --order $d --output_suffix o$d
done
```

### Table 7.3 (Temperature)
```bash
for t in 0.5 1.0 2.0; do
    python run_all_4_padre.py --temperature $t --output_suffix t$t
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

| Model | Acc |
|---|---|
| DistilBERT | 90.37% ± 1% |
| RoBERTa | 64.56% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-padre`
- **Metrics:** `results/unified_padre/*/summary.csv`

---