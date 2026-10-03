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
python run_all_4_orion.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 95\% \
    --svd_max_rank 64 \
    --rff_features 64 \
    --recovery_steps 50
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-orion",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_orion.py --models distilbert,roberta,gpt2
```

### SVD Energy Ablation

```bash
for e in 0.90 0.95 0.99; do
    python run_all_4_orion.py --svd_energy $e --output_suffix e$e
done
```

### Head Split Ablation

```bash
for split in "half" "23_13" "34_14" "all_kernel" "all_full"; do
    python run_all_4_orion.py --head_split $split --output_suffix $split
done
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Reduction |
|---|---|---|
| DistilBERT | 55.50% ± 2% | 48.2\%× |
| RoBERTa | 50.92% ± 3% | 45.1\%× |
| GPT-2 | 50.92% ± 2% | 46.0\%× |

## 12.7 Diagnostics

### Check Parameter Count

```python
for name, mod in model.named_modules():
    if hasattr(mod, 'down') and hasattr(mod, 'up'):
        orig = mod.in_features * mod.out_features
        new = mod.down.weight.numel() + mod.up.weight.numel()
        print(f"{name}: {orig} → {new} ({orig/new:.1f}×)")
```

### Check Head Split

```python
print(f"Kernel heads: {model.num_kernel_heads}")
print(f"Full heads: {model.num_full_heads}")
```

### Check rff_proj Persistence

```python
sd = model.state_dict()
assert any('rff_proj' in k for k in sd), "rff_proj NOT in state_dict!"  # This will FAIL
```

## 12.8 Known Issues

- **`copy_weights` raises** on UniversalAdapter failure.
- **`rff_proj` not persisted** in state_dict.
- **Softpick zeros** on masked rows.
- **Slower than baseline** at short sequences.

## 12.9 Data Availability

- Code: `framework/attentions/orion.py`
- Metrics: `results/unified_orion/*/summary.csv`
- Logs: `benchmark_logs/orion_*.log`

---