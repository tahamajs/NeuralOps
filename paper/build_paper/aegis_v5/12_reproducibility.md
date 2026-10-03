# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running AEGIS-V5

```bash
python run_all_4_aegis_v5.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 0.95 \
    --max_rank 384 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,aegis_v5 \
    --svd_energy 0.95 \
    --max_rank 384
```

## 12.4 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_aegis_v5.py
```

### Table 7.5 (Rank Ablation)
```bash
for r in 32 64 128 256 384; do
    python run_all_4_aegis_v5.py - max_rank $r - output_suffix r$r
done
```

### Table 7.6 (Head Split)
```bash
python run_all_4_aegis_v5.py - num_local_heads 12 - num_global_heads 0 - output_suffix alllocal
python run_all_4_aegis_v5.py - num_local_heads 8 - num_global_heads 4 - output_suffix default
python run_all_4_aegis_v5.py - num_local_heads 0 - num_global_heads 12 - output_suffix allglobal
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
| DistilBERT | 83.72% ± 1% |
| RoBERTa | 86.47% ± 1% |
| GPT-2 | 69.50% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-aegis_v5`
- **Metrics:** `results/unified_aegis_v5/*/summary.csv`

---