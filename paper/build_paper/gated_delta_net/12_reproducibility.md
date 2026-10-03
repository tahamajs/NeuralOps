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
source .venv/bin/activate  # or .venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running the Benchmark

```bash
# Full run (may timeout on RoBERTa/GPT-2)
python run_all_4_gated_delta_net.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8

# DistilBERT only (fastest)
python run_all_4_gated_delta_net.py --models distilbert
```

## 12.3 Expected Runtime

| Model | Expected runtime |
|---|---|
| DistilBERT | ~2.3 s min |
| RoBERTa | OOM or >60 min |
| GPT-2 | OOM or >60 min |

## 12.4 Per-Step Timing Measurement

To reproduce our timing results:

```python
import time

# Wrap 5 training steps
start = time.time()
for _ in range(5):
    ...  # training step
elapsed = time.time() - start
per_step = elapsed / 5
print(f"Per-step: {per_step:.2f} s")
```

## 12.5 Common Failure Modes

### 12.5.1 RoBERTa/GPT-2 OOM

Reduce batch size:

```bash
python run_all_4_gated_delta_net.py --batch_size 2 --recovery_steps 10
```

### 12.5.2 Timeout

Increase timeout:

```bash
python run_all_4_gated_delta_net.py --timeout 7200
```

### 12.5.3 Chunk Size Mismatch

If `chunk_size` does not divide `L`, padding is applied. Verify:

```python
assert L % chunk_size == 0 or padding_ok
```

## 12.6 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Verification

### Check State Update

```python
# Single-token update
k = torch.randn(1, 1, 1, 64)
v = torch.randn(1, 1, 1, 64)
S = torch.zeros(1, 1, 64, 64)
beta = torch.tensor([[[0.9]]])
alpha = torch.tensor([[[0.5]]])

from framework.attentions.gated_delta_net import _delta_chunk_forward

y, S_new = _delta_chunk_forward(S, k, k, v, beta, alpha)
# Verify: S_new ≈ beta * S + alpha * k^T (v - k S)
```

## 12.8 Data Availability

- Code: `framework/attentions/gated_delta_net.py`
- Metrics: `results/unified_gated_delta_net/*/summary.csv`
- Logs: `benchmark_logs/gated_delta_net_*.log`

## 12.9 Known Issues

- **Slow:** ~90s per 5 steps on DistilBERT.
- **OOM:** RoBERTa, GPT-2 at batch_size=8.
- **Windows:** `cp1256` codec issues.

## 12.10 Recommended Fix

For a fast implementation, use:

- `flash-linear-attention` library (Triton kernels)
- `mamba-ssm` (for Mamba-like state spaces)
- Custom Triton kernel (see Section 5.10)

## 12.11 Reproduction Checklist

- [ ] GPU printed
- [ ] `chunk_size` logged
- [ ] `use_gradient_checkpointing` logged
- [ ] Per-step timing measured
- [ ] Peak VRAM measured
- [ ] OOM errors captured

---