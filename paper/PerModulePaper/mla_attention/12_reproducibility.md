<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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
| numpy | {{NUMPY_VERSION}} |

## 12.3 Running MLA

```bash
python run_all_4_mla.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}} \
    --batch_size {{BATCH_SIZE}}
```

**Runtime:** ~{{RUNTIME}} minutes.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_mla/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,mla \
    --rank {{RANK}} \
    --lora_rank {{LORA_RANK}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_mla.py --models distilbert,roberta,gpt2
```

### Table 7.7 (Latent dimension)
```bash
for r in 64 128 256 512; do
    python run_all_4_mla.py --rank $r --output_suffix r$r
done
```

### Table 7.8 (LoRA rank)
```bash
for lr in 4 8 16 32; do
    python run_all_4_mla.py --lora_rank $lr --output_suffix lr$lr
done
```

### Table 7.10 (Normalization)
```bash
python run_all_4_mla.py --normalize_sv true --output_suffix norm
python run_all_4_mla.py --normalize_sv false --output_suffix nonorm
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

| Model | Acc | KV ratio |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | {{D_KV_RATIO}}× |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_KV_RATIO}}× |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_KV_RATIO}}× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-mla",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-mla`
- Metrics: `results/unified_mla/*/summary.csv`
- Logs: `benchmark_logs/mla_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- MLA is single-layer injection only; whole-model traversal is caller's 
  responsibility
- Original DeepSeek-V2 not tested

## 12.11 Original Paper

MLA is from DeepSeek-AI (2024), arXiv:2405.04434. We recommend reading 
the original for the algorithmic details.
```

---