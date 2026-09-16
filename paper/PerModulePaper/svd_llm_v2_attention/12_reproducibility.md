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

## 12.3 Running SVD-LLM V2

```bash
python run_all_4_svd_llm_v2.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --energy {{ENERGY}} \
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
    --output_dir results/unified_svd_llm_v2/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,svd_llm_v2 \
    --rank {{RANK}} \
    --energy {{ENERGY}} \
    --lora_rank {{LORA_RANK}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_svd_llm_v2.py --models distilbert,roberta,gpt2
```

### Table 7.7 (Rank cap ablation)
```bash
for r in 32 64 128 192; do
    python run_all_4_svd_llm_v2.py --rank $r --output_suffix r$r
done
```

### Table 7.8 (Energy ablation)
```bash
for eta in 0.90 0.95 0.99 0.999; do
    python run_all_4_svd_llm_v2.py --energy $eta --output_suffix e$eta
done
```

### Table 7.10 (Explicit vs symmetric)
```bash
python run_all_4_svd_llm_v2.py --factorization explicit --output_suffix exp
python run_all_4_svd_llm_v2.py --factorization symmetric --output_suffix sym
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
| DistilBERT | {{D_ACC}}% ± 1% | {{D_RATIO}}× |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_RATIO}}× |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_RATIO}}× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-svd_llm_v2",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-svd_llm_v2`
- Metrics: `results/unified_svd_llm_v2/*/summary.csv`
- Logs: `benchmark_logs/svd_llm_v2_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Explicit storage may be FP16-sensitive; use symmetric if needed

---