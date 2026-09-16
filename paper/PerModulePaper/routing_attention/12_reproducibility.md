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
python run_all_4_routing.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters {{NUM_CLUSTERS}} \
    --n_iters {{N_ITERS}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

**Runtime:** ~{{RUNTIME}} minutes.

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"{{HF_ORG}}/svd-linear-attention-{model}-routing",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1

```bash
python run_all_4_routing.py --models distilbert,roberta,gpt2
```

### Table 7.4 (num_clusters ablation)

```bash
for K in 4 8 16 32; do
    python run_all_4_routing.py --num_clusters $K --output_suffix K$K
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

## 12.7 Known Issues

- Windows `cp1256` codec errors: set `PYTHONIOENCODING=utf-8`.
- FP16 attention may overflow for $L > 512$.
- Padding distorts k-means clustering.

## 12.8 Data Availability

- Code: `framework/attentions/routing_attention.py`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-routing`
- Metrics: `results/unified_routing/*/summary.csv`
- Logs: `benchmark_logs/routing_*.log`

---