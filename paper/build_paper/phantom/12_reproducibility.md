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
| numpy | 1.0.0 |

## 12.3 Running PHANTOM

```bash
python run_all_4_phantom.py \
    --models distilbert,roberta,gpt2 \
    --cur_rank 32 \
    --lsh_rounds 4 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes on RTX 4070.

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_phantom/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,phantom \
    --cur_rank 32 \
    --lsh_rounds 4 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_phantom.py --models distilbert,roberta,gpt2
```

### Table 7.6 (CUR rank)

```bash
for r in 8 16 32 64; do
    python run_all_4_phantom.py --cur_rank $r --output_suffix r$r
done
```

### Table 7.7 (LSH rounds)

```bash
for rounds in 1 2 4 8; do
    python run_all_4_phantom.py --lsh_rounds $rounds --output_suffix lsh$rounds
done
```

### Table 7.8 (LSH bits)

```bash
for bits in 2 3 4 6; do
    python run_all_4_phantom.py --lsh_bucket_size $bits --output_suffix k$bits
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
| DistilBERT | 72.59% ± 1% | 50\%× |
| RoBERTa | 50.92% ± 1% | 2.1\times× |
| GPT-2 | 50.92% ± 1% | 2.1\times× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-phantom`
- **Metrics:** `results/unified_phantom/*/summary.csv`

## 12.9 Known Issues

- Windows: set `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Batch size 4 required for 12-layer models on 8 GB GPU

---