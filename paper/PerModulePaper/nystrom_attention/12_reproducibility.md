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

## 12.3 Running Nyströmformer

### Short Sequence (N=128)

```bash
python run_all_4_nystrom.py \
    --models distilbert,roberta,gpt2 \
    --seq_len 128 \
    --num_landmarks 32 \
    --recovery_steps 50 \
    --lora_rank 8
```

### Long Sequence (N=512)

```bash
python run_all_4_nystrom.py \
    --models distilbert,roberta \
    --seq_len 512 \
    --num_landmarks 32
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_nystrom/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,nystrom \
    --num_landmarks 32
```

## 12.5 Reproducing Tables

### Table 7.1 (Short sequence)

```bash
python run_all_4_nystrom.py --models distilbert,roberta,gpt2
```

### Table 7.2 (Long sequence)

```bash
for m in 16 32 64 128; do
    python run_all_4_nystrom.py --models distilbert --seq_len 512 --num_landmarks $m --output_suffix m$m
done
```

### Table 7.4 (Landmark ablation)

```bash
for m in 16 32 64 128; do
    python run_all_4_nystrom.py --num_landmarks $m --output_suffix m$m
done
```

### Table 7.5 (Pinv iterations)

```bash
for it in 1 2 3 5; do
    python run_all_4_nystrom.py --pinv_iterations $it --output_suffix it$it
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

| Model | N=128 | N=512 |
|---|---|---|
| DistilBERT | Dense (baseline) | Nyström (86.2× speedup) |
| RoBERTa | Dense (baseline) | Nyström (86.2× speedup) |
| GPT-2 | Dense causal | Dense causal |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-nystrom",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-nystrom`
- Metrics: `results/unified_nystrom/*/summary.csv`
- Logs: `benchmark_logs/nystrom_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Nyström branch inactive for $N \leq 128$
- Causal decoders always use dense fallback

## 12.11 Original Paper

Nyströmformer is from Xiong et al. (2021), arXiv:2102.03902.

---