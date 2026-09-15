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

## 12.3 Running Dobi-SVD

```bash
python run_all_4_dobi.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --init_gate_val {{INIT_GATE}} \
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
    --output_dir results/unified_dobi/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,dobi \
    --rank {{RANK}} \
    --init_gate_val {{INIT_GATE}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_dobi.py --models distilbert,roberta,gpt2
```

### Table 7.6 (Rank ablation)
```bash
for r in 64 128 192 256; do
    python run_all_4_dobi.py --rank $r --output_suffix r$r
done
```

### Table 7.7 (Init gate ablation)
```bash
for g in 1.0 2.5 4.0 5.0; do
    python run_all_4_dobi.py --init_gate_val $g --output_suffix g$g
done
```

### Table 7.8 (Pruning threshold)
```bash
python post_hoc_prune.py --tau 0.1 --output_suffix tau1
python post_hoc_prune.py --tau 0.3 --output_suffix tau3
python post_hoc_prune.py --tau 0.5 --output_suffix tau5
python post_hoc_prune.py --tau 0.7 --output_suffix tau7
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

| Model | Acc | Eff. rank |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | {{D_REFF}} |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_EFF}} |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_EFF}} |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-dobi",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-dobi`
- Metrics: `results/unified_dobi/*/summary.csv`
- Logs: `benchmark_logs/dobi_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Gates require at least 30 recovery steps to converge
- Post-hoc pruning requires a separate script

---
