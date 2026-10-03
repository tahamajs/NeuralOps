<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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

## 12.3 Running ASVD

```bash
python run_all_4_asvd.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --energy_threshold 0.95 \
    --error_threshold 0.20 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,asvd \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_asvd.py --models distilbert,roberta,gpt2
```

### Table 7.6 (Rank)
```bash
for r in 16 32 64 128; do
    python run_all_4_asvd.py --rank $r --output_suffix r$r
done
```

### Table 7.7 (Energy threshold)
```bash
for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_asvd.py --energy_threshold $e --output_suffix e$e
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
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 90.37% | 5.5× |
| RoBERTa | 77.75% | 5.0× |
| GPT-2 | 85.78% | 5.0× |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `Hooshaai/svd-linear-attention-*-asvd`
- Metrics: `results/unified_asvd/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- `raise` may abort injection on unusual sources

---