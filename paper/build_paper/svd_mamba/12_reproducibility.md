<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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

## 12.3 Running SVD-Mamba

```bash
python run_all_4_svd_mamba.py \
    --models distilbert,roberta,gpt2 \
    --d_state Optimal \
    --d_conv 15 steps \
    --global_rank 32 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,svd_mamba \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2
```

### Table 7.4 (d_state)
```bash
for ds in 8 16 32 64; do
    python run_all_4_svd_mamba.py --d_state $ds --output_suffix ds$ds
done
```

### Table 7.7 (gate bias)
```bash
for gb in -10 -7 -5 0; do
    python run_all_4_svd_mamba.py --gate_bias $gb --output_suffix gb$gb
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

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | 50.92% | +48.5\%% |
| RoBERTa | 50.92% | +48.5\%% |
| GPT-2 | 50.23% | +48.5\%% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-svd_mamba`
- Metrics: `results/unified_svd_mamba/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention returns input (no-op)
- Complex forward pass with Python loop over chunks
- Random state-space initialization affects results

---