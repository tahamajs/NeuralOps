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
python run_all_4_decay_linear.py \
    --models distilbert,roberta,gpt2 \
    --chunk_size {{CHUNK_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,decay_linear \
    --chunk_size {{CHUNK_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_decay_linear.py
```

### Table 7.5 (Chunk Size)
```bash
for c in 8 16 32 64 128; do
    python run_all_4_decay_linear.py --chunk_size $c --output_suffix c$c
done
```

### Table 7.6 (Log-Decay Range)
```bash
python run_all_4_decay_linear.py --log_decay_min -2.0 --output_suffix r1
python run_all_4_decay_linear.py --log_decay_min -1.0 --output_suffix r2
python run_all_4_decay_linear.py --log_decay_min -0.5 --output_suffix r3
python run_all_4_decay_linear.py --log_decay_min 0.0 --log_decay_max 0.5 --output_suffix r4
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

| Model | Acc | State mem |
|---|---|---|
| DistilBERT | {{D_ACC}}% | 16 KB |
| RoBERTa | {{R_ACC}}% | 16 KB |
| GPT-2 | {{G_ACC}}% | 16 KB |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-decay_linear`.
- **Metrics:** `results/unified_decay_linear/*/summary.csv`.

---