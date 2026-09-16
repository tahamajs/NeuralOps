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

## 12.2 Running

```bash
python run_all_4_salt.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-salt",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1

```bash
python run_all_4_salt.py --models distilbert,roberta,gpt2
```

### Table 7.4 (rank ablation)

```bash
for r in 4 8 16 32 64; do
    python run_all_4_salt.py --rank $r --output_suffix r$r
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy |
|---|---|
| DistilBERT | {{D_ACC}}% ± 1% |
| RoBERTa | {{R_ACC}}% ± 1% |
| GPT-2 | {{G_ACC}}% ± 1% |

## 12.7 Storage Verification

To verify storage reduction:

```python
import torch
model = torch.load("weights.pt")
total = sum(p.numel() for p in model.values())
print(f"Total parameters: {total}")
# Expected: ~5.6M for DistilBERT (vs 67M baseline)
```

## 12.8 Known Issues

- RuntimeError on missing projections
- RuntimeError on zero norm
- Unused `output_scale`
- Slow inference (no fused forward)

## 12.9 Data Availability

- Code: `framework/attentions/salt_attention.py`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-salt`
- Metrics: `results/unified_salt/*/summary.csv`
- Logs: `benchmark_logs/salt_*.log`

---