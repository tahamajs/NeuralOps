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
| numpy | 1.26.4 |

## 12.3 Running AdaSplash

```bash
python run_all_4_adasplash.py \
    --models distilbert,roberta,gpt2 \
    --alpha 1.5 \
    --splash_radius 32 \
    --lora_rank 16 \
    --recovery_steps 50
```

**Runtime:** ~20 minutes.

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"Hooshaai/svd-linear-attention-{model}-adasplash",
        filename="weights.pt",
    )
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_adasplash.py - models distilbert,roberta,gpt2
```

### Table 7.5 (α ablation)
```bash
for alpha in 1.0 1.2 1.5 1.8 2.0; do
    python run_all_4_adasplash.py - alpha $alpha - output_suffix a$alpha
done
```

### Table 7.6 (Splash radius)
```bash
for W in 8 16 32 64 128; do
    python run_all_4_adasplash.py - splash_radius $W - output_suffix W$W
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
```

## 12.7 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 77.18% ± 1% |
| RoBERTa | 67.09% ± 1% |
| GPT-2 | 79.36% ± 1% |

## 12.8 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `Hooshaai/svd-linear-attention-*-adasplash`
- Metrics: `results/unified_adasplash/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Entmax bisection: 25 iterations
