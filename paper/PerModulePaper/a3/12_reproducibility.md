<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Full reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |
| numpy | {{NUMPY_VERSION}} |
| pandas | {{PANDAS_VERSION}} |

## 12.3 Running A3

### Full Benchmark

```bash
python run_all_4_a3.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}} \
    --batch_size {{BATCH_SIZE}}
```

**Runtime:** ~{{RUNTIME}} minutes on RTX 4070.

### Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --dataset_dir Codes/framework/data/glue_data/sst2 \
    --output_dir results/unified_a3/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,a3 \
    --batch_size {{BATCH_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --learning_rate {{LR}} \
    --lora_rank {{LORA_RANK}} \
    --rank {{RANK}} \
    --precision fp16
```

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-a3",
        filename="weights.pt",
    )
    print(f"Downloaded {model} to {path}")
```

## 12.5 Reproducing Tables

### Table 7.1 (Main Results)

```bash
python run_all_4_a3.py --models distilbert,roberta,gpt2
# → results/unified_a3/{model}/summary.csv
```

### Table 7.6 (Rank Ablation)

```bash
for rank in 32 64 128 192 384; do
    python run_all_4_a3.py --models distilbert,roberta,gpt2 --rank $rank
done
```

### Table 7.7 (Fisher Ablation)

```bash
python run_all_4_a3.py --use_fisher true  --output_suffix fisher
python run_all_4_a3.py --use_fisher false --output_suffix nofisher
```

### Table 7.8 (LN Ablation)

```bash
python run_all_4_a3.py --recalibrate_ln true  --output_suffix ln
python run_all_4_a3.py --recalibrate_ln false --output_suffix noln
```

## 12.6 Random Seeds

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 12.7 Expected Results

| Model | Accuracy | Ratio |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | {{D_RATIO}}× |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_RATIO}}× |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_RATIO}}× |

Deviations >2% suggest environmental differences.

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets` library
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-a3`
- **Metrics:** `results/unified_a3/*/summary.csv`
- **Logs:** `benchmark_logs/a3_*.log`

## 12.9 Reproducibility Checks

Before running, verify:

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('CUDA ver:', torch.version.cuda)
"
```

## 12.10 Known Issues

- Windows: `cp1256` codec issues; set `PYTHONIOENCODING=utf-8`
- CUDA version: >= 11.8 recommended
- Checkpoint paths: some baselines may need to be re-downloaded
