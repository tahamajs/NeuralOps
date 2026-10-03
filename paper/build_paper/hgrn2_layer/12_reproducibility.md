# 12. Reproducibility

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

## 12.3 Running

```bash
python run_all_4_hgrn2.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,hgrn2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_hgrn2.py
```

### Table 7.4 (Chunk Ablation)
```bash
python run_all_4_hgrn2.py --chunk_size 8
python run_all_4_hgrn2.py --chunk_size 16
python run_all_4_hgrn2.py --chunk_size 32
python run_all_4_hgrn2.py --chunk_size 64
```

### Table 7.5 (Bidirectional)
```bash
python run_all_4_hgrn2.py --is_decoder false  # bidirectional
python run_all_4_hgrn2.py --is_decoder true   # causal
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

| Model | Acc | Params Δ |
|---|---|---|
| DistilBERT | 62.96% | +48.5\%% |
| RoBERTa | 52.06% | +48.5\%% |
| GPT-2 | 71.44% | +48.5\%% |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-hgrn2`.
- **Metrics:** `results/unified_hgrn2/*/summary.csv`.

---