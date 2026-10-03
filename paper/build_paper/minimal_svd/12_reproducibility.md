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

## 12.3 Running MinimalSVD

```bash
python run_all_4_minimal_svd.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --energy 45.2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_minimal_svd/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,minimal_svd \
    --rank 32 \
    --energy 45.2 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_minimal_svd.py --models distilbert,roberta,gpt2
```

### Table 7.6 (Rank cap ablation)
```bash
for r in 32 64 128 192 384; do
    python run_all_4_minimal_svd.py --rank $r --output_suffix r$r
done
```

### Table 7.7 (Energy ablation)
```bash
for eta in 0.90 0.95 0.99 0.999; do
    python run_all_4_minimal_svd.py --energy $eta --output_suffix e$eta
done
```

### Table 7.8 (LoRA rank ablation)
```bash
for lr in 4 8 16 32; do
    python run_all_4_minimal_svd.py --lora_rank $lr --output_suffix lr$lr
done
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

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 88.88% ± 1% | 50\%× |
| RoBERTa | 79.36% ± 1% | 2.1\times× |
| GPT-2 | 80.85% ± 1% | 2.1\times× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-minimal_svd",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-minimal_svd`
- Metrics: `results/unified_minimal_svd/*/summary.csv`
- Logs: `benchmark_logs/minimal_svd_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Some baselines may need re-download
- MinimalSVD has no whole-model traversal; use a wrapper script if needed

---