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

## 12.3 Running UnifiedRoPE

```bash
python run_all_4_unified_rope.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_unified_rope/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,unified_rope \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_unified_rope.py --models distilbert,roberta,gpt2
```

### Table 7.6 (RoPE base ablation)

```bash
for b in 5000 10000 100000; do
    python run_all_4_unified_rope.py --base $b --output_suffix b$b
done
```

### Table 7.7 (Rotary dimension)

```bash
for rd in 32 64 128; do
    python run_all_4_unified_rope.py --rotary_dim $rd --output_suffix rd$rd
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

| Model | Acc | Params |
|---|---|---|
| DistilBERT | 87.96% ± 1% | 1.00× |
| RoBERTa | 64.33% ± 1% | 1.00× |
| GPT-2 | 57.45% ± 1% | 1.00× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-unified_rope",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-unified_rope`
- Metrics: `results/unified_unified_rope/*/summary.csv`
- Logs: `benchmark_logs/unified_rope_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- RoPE cache is non-persistent, so checkpoints do not include it
- Encoder models need recovery fine-tuning

---