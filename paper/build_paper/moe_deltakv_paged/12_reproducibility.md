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
python run_all_4_moe_deltakv_paged.py \
    --models distilbert,roberta,gpt2 \
    --page_size 64 \
    --k_experts 4 \
    --svd_rank 16 \
    --recovery_steps 50
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,moe_deltakv_paged \
    --page_size 64 \
    --k_experts 4
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_moe_deltakv_paged.py
```

### Table 7.4 (k_experts ablation)
```bash
for k in 1 2 4 8; do
    python run_all_4_moe_deltakv_paged.py --k_experts $k --output_suffix k$k
done
```

### Table 7.6 (rank ablation)
```bash
for r in 4 8 16 32 64; do
    python run_all_4_moe_deltakv_paged.py --svd_rank $r --output_suffix r$r
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 86.47% ± 1% |
| RoBERTa | 59.75% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-moe_deltakv_paged`
- **Metrics:** `results/unified_moe_deltakv_paged/*/summary.csv`

---