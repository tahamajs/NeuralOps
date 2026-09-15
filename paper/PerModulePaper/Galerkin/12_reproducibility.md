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
source .venv/bin/activate  # or .venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running the Benchmark

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-galerkin",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1 (Main Results)

```bash
python run_all_4_galerkin.py --models distilbert,roberta,gpt2
```

### Table 7.4 (Rank Ablation)

```bash
for rank in 8 16 32 64 128; do
    python run_all_4_galerkin.py --rank $rank --output_suffix r$rank
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Status |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 2% | {{D_STATUS}} |
| RoBERTa | {{R_ACC}}% ± 2% | {{R_STATUS}} |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_STATUS}} |

Deviations >2% suggest environment differences.

## 12.7 Diagnostic Checks

### Verify Galerkin Path is Taken (Encoders)

```python
# Add to forward:
logger.info(f"Galerkin path: {not self.is_decoder}")

# Expected for DistilBERT: True
# Expected for GPT-2: False
```

### Verify Basis Orthonormality

```python
Phi = model.get_basis(N=128, r=64)
err = (Phi.T @ Phi - torch.eye(64)).abs().max().item()
print(f"Orthonormality error: {err}")  # Should be < 1e-5
```

## 12.8 Data Availability

- Code: `framework/attentions/galerkin.py`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-galerkin`
- Metrics: `results/unified_galerkin/*/summary.csv`
- Logs: `benchmark_logs/galerkin_*.log`

## 12.9 Known Issues

- Windows: `cp1256` codec issues; set `PYTHONIOENCODING=utf-8`.
- FP16 basis QR occasionally produces NaN; use FP32 if issues arise.

---
