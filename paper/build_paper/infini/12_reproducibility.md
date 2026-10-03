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

## 12.3 Running Infini-Attention

```bash
python run_all_4_infini.py \
    --models distilbert,roberta,gpt2 \
    --compress_dim 64 \
    --local_window 128 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,infini \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_infini.py --models distilbert,roberta,gpt2
```

### Table 7.4 (compress_dim)
```bash
for dc in 32 64 128 256; do
    python run_all_4_infini.py --compress_dim $dc --output_suffix dc$dc
done
```

### Table 7.5 (window)
```bash
for w in 32 64 128 256; do
    python run_all_4_infini.py --local_window $w --output_suffix w$w
done
```

### Table 7.6 (β init)
```bash
python run_all_4_infini.py --beta_init -10
python run_all_4_infini.py --beta_init -5
python run_all_4_infini.py --beta_init -2
python run_all_4_infini.py --beta_init 0
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | 90.25% | +48.5\%% |
| RoBERTa | 56.54% | +48.5\%% |
| GPT-2 | 51.26% | +48.5\%% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-infini`
- Metrics: `results/unified_infini/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- Cross-attention not supported
- RoBERTa may need `is_nested_encoder` adjustment

---