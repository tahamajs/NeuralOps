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
python run_all_4_hgrn2_gated_deltanet.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

**Warning:** RoBERTa may OOM on 8 GB GPUs. Use `--batch_size 4` if needed.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,hgrn2_gated_deltanet \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_hgrn2_gated_deltanet.py
```

### Table 7.3 (Speed vs L)
```bash
python scripts/benchmark_inference.py --seq_lens 128 256 512 1024 2048
```

### Table 7.6 (Branch Ablation)
```bash
python run_all_4_hgrn2_gated_deltanet.py --use_hgrn true --use_deltanet false
python run_all_4_hgrn2_gated_deltanet.py --use_hgrn false --use_deltanet true
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

| Model | Acc | Params Δ | Speed @ L=128 |
|---|---|---|---|
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% | {{SLOWDOWN}}× |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% | {{R_SLOWDOWN}}× |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% | {{G_SLOWDOWN}}× |

## 12.8 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-hgrn2_gated_deltanet`
- **Metrics:** `results/unified_hgrn2_gated_deltanet/*/summary.csv`

---
