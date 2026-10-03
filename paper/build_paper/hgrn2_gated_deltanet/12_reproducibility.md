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
python run_all_4_hgrn2_gated_deltanet.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

**Warning:** RoBERTa may OOM on 8 GB GPUs. Use `--batch_size 4` if needed.

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,hgrn2_gated_deltanet \
    --recovery_steps 50 \
    --lora_rank 8
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
| DistilBERT | 53.56% | +48.5\%% | 1.0x× |
| RoBERTa | 52.98% | +48.5\%% | 84.5× |
| GPT-2 | 51.03% | +48.5\%% | 84.5× |

## 12.8 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-hgrn2_gated_deltanet`
- **Metrics:** `results/unified_hgrn2_gated_deltanet/*/summary.csv`

---