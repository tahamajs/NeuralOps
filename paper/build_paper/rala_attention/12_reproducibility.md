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
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |
| numpy | 1.0.0 |

## 12.3 Running

```bash
python run_all_4_rala.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --rala_rank 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-rala",
        filename="weights.pt",
    )
    print(f"Downloaded {model}")
```

## 12.5 Reproducing Tables

### Table 7.1

```bash
python run_all_4_rala.py --models distilbert,roberta,gpt2
```

### Table 7.4 (rank ablation)

```bash
for rank in 1 2 4 8 16; do
    python run_all_4_rala.py --rala_rank $rank --output_suffix rank$rank
done
```

### Table 7.5 (alpha init)

```bash
for a in 0.0 0.1 1.0; do
    python run_all_4_rala.py --alpha_init $a --output_suffix alpha$a
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Accuracy |
|---|---|
| DistilBERT | 60.09% ± 1% |
| RoBERTa | 50.80% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.8 Known Issues

- Windows `cp1256` codec errors: set `PYTHONIOENCODING=utf-8`.
- FP16 prefix scan may lose precision for $L > 512$.
- Qwen3.5 DeltaNet layers are skipped.

## 12.9 Data Availability

- Code: `framework/attentions/rala_linear_attention.py`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-rala`
- Metrics: `results/unified_rala/*/summary.csv`
- Logs: `benchmark_logs/rala_*.log`

---