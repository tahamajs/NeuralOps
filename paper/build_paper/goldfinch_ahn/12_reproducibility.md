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
python run_all_4_goldfinch_ahn.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --window_size 128
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-goldfinch_ahn",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_goldfinch_ahn.py --models distilbert,roberta,gpt2
```

### Window Ablation

```bash
for W in 16 32 64 128; do
    python run_all_4_goldfinch_ahn.py --window_size $W --output_suffix w$W
done
```

### Gate Init Bias Ablation

```bash
for b in 0.0 1.0 1.8 3.0; do
    python run_all_4_goldfinch_ahn.py --gate_init_bias $b --output_suffix b$b
done
```

### Preservation Gate Ablation

```bash
for g in 0.0 1.0 2.0 5.0; do
    python run_all_4_goldfinch_ahn.py --output_gate_init $g --output_suffix ag$g
done
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
| GPT-2 | 50.92% ± 2% | PASS |

## 12.7 Diagnostics

### Verify Window Mask

```python
import torch

L = 16
W = 4
idx = torch.arange(L)
dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
M = dist <= W
# Check band structure
assert M[0, 0] == True
assert M[0, W + 1] == False
```

### Verify Preservation Gate

```python
import torch

alpha = torch.sigmoid(torch.tensor(2.0))
# alpha ≈ 0.88
assert 0.85 < alpha.item() < 0.90
```

### Verify Gate Mean

```python
x = torch.randn(2, 8, 768)
gate_raw = torch.sigmoid(torch.nn.Linear(768, 12)(x))  # (2, 8, 12)
gate = gate_raw.mean(dim=-1, keepdim=True)  # (2, 8, 1)
# Confirm scalar per token
assert gate.shape == (2, 8, 1)
```

## 12.8 Known Issues

- **Fake long-memory branch.** `GatedDeltaNetLayer` is `nn.Linear`.
- **Mean-pooled gate.** Loses per-head info.
- **`output_scale` redundancy** in `LoRALinear`.

## 12.9 Data Availability

- Code: `framework/attentions/goldfinch_ahn.py`
- Metrics: `results/unified_goldfinch_ahn/*/summary.csv`
- Logs: `benchmark_logs/goldfinch_ahn_*.log`

---