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
python run_all_4_performer.py \
    --models distilbert,roberta,gpt2 \
    --num_features 64 \
    --recovery_steps 50
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-performer",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_performer.py --models distilbert,roberta,gpt2
```

### Feature Ablation

```bash
for F in 32 64 128 256; do
    python run_all_4_performer.py --num_features $F --output_suffix f$F
done
```

### Kernel Type Ablation

```bash
python run_all_4_performer.py --kernel_type favor_plus --output_suffix favor
python run_all_4_performer.py --kernel_type elu --output_suffix elu
python run_all_4_performer.py --kernel_type relu --output_suffix relu
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Status |
|---|---|---|
| DistilBERT | 50.92% ± 2% | PASS |
| RoBERTa | 50.92% ± 3% | PASS |
| GPT-2 | 77.5% ± 2% | PASS |

## 12.7 Diagnostics

### Verify Feature Positivity

```python
Q_feat = model._feature_map(Q)
assert (Q_feat >= 0).all(), "Feature map produced negative values!"
```

### Verify Omega Persistence

```python
sd = model.state_dict()
assert any("omega" in k for k in sd), "omega NOT persisted!"
```

### Verify Causal Path

```python
x = torch.randn(1, 128, 768, requires_grad=True)
y = model(x)
y[0, 50].sum().backward()
assert x.grad[0, 51:].abs().max() < 1e-6
```

## 12.8 Known Issues

- **`inject_performer` raises** instead of returning None.
- **`output_scale`** new parameter.
- **Cross-attention not supported.**

## 12.9 Data Availability

- Code: `framework/attentions/performer.py`
- Metrics: `results/unified_performer/*/summary.csv`
- Logs: `benchmark_logs/performer_*.log`

---