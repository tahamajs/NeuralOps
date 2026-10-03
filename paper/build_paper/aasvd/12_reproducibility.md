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
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |
| numpy | 1.26.4 |

## 12.3 Running AASVD

```bash
python run_all_4_aasvd.py \
    --models distilbert,roberta,gpt2 \
    --rank 288 \
    --recovery_steps 50 \
    --lora_rank 16 \
    --batch_size 8
```

**Runtime:** ~20 minutes on RTX 4070.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_aasvd/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,aasvd \
    --rank 288 \
    --lora_rank 16
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_aasvd.py - models distilbert,roberta,gpt2
```

### Table 7.6 (Rank Ablation)

```bash
for rank in 32 64 128 256; do
    python run_all_4_aasvd.py - rank $rank - output_suffix r$rank
done
```

### Table 7.7 (Covariance Ablation)

```bash
python run_all_4_aasvd.py - use_covariance true - output_suffix fullcov
python run_all_4_aasvd.py - use_covariance false - output_suffix noweight
```

### Table 7.8 (EMA β Ablation)

```bash
for beta in 0.5 0.7 0.9 0.95 0.99; do
    python run_all_4_aasvd.py - ema_beta $beta - output_suffix b$beta
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
| DistilBERT | 89.45% ± 1% | 4.0× |
| RoBERTa | 88.07% ± 1% | 3.5× |
| GPT-2 | 80.73% ± 1% | 3.5× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-aasvd`
- **Metrics:** `results/unified_aasvd/*/summary.csv`
- **Calibration corpus:** bundled in `aasvd_compressor.py`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- CUDA ≥ 11.8 for FP16 SVD
- Some baselines may need re-download
