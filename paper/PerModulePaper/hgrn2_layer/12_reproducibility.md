# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |

## 12.3 Running

```bash
python run_all_4_hgrn2.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,hgrn2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
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
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-hgrn2`.
- **Metrics:** `results/unified_hgrn2/*/summary.csv`.

---