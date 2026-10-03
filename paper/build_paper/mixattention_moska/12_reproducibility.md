<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_fasa.py \
    --models distilbert,roberta,gpt2 \
    --fasa_keep_ratio 0.5 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-fasa",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_fasa.py --models distilbert,roberta,gpt2
```

### Keep Ratio Ablation

```bash
for r in 0.25 0.5 0.75 1.0; do
    python run_all_4_fasa.py --fasa_keep_ratio $r --output_suffix k$r
done
```

### Importance Method Ablation

```bash
python run_all_4_fasa.py --importance_method var --output_suffix var
python run_all_4_fasa.py --importance_method l2 --output_suffix l2
```

### CLS Protection Ablation

```bash
python run_all_4_fasa.py --protect_first_token true --output_suffix protect
python run_all_4_fasa.py --protect_first_token false --output_suffix noprotect
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Path |
|---|---|---|
| DistilBERT | 90.94% ± 2% | Pruned |
| RoBERTa | 86.24% ± 3% | Pruned |
| GPT-2 | 87.27% ± 1% | No-prune |

## 12.7 Diagnostics

### Verify Pruning Applies

```python
# Count actual kept tokens
kept = int(L * fasa_keep_ratio)
print(f"Expected kept tokens: {max(8, min(L, kept))}")
```

### Verify CLS Pinning

```python
# Index 0 should always be in the top-k
imp = k.float().var(dim=-1).mean(dim=1)
imp[:, 0] = 1e9
_, idx = torch.topk(imp, keep, dim=-1)
assert 0 in idx[0].tolist()
```

### Verify Decoder Skips Pruning

```python
# For GPT-2, is_causal should be True
assert model.is_decoder == True
```

## 12.8 Known Issues

- **Decoder path is a no-op** for GPT-2.
- **Slower than baseline** at short sequences.
- **Aggressive keep ratio** (0.5) causes accuracy drop.

## 12.9 Data Availability

- Code: `framework/attentions/fasa.py`
- Metrics: `results/unified_fasa/*/summary.csv`
- Logs: `benchmark_logs/fasa_*.log`

---