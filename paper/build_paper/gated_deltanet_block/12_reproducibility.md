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
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

### Path A (Full Attention)

```bash
python run_all_4_gated_deltanet_block.py \
    --models distilbert \
    --use_full_attention true \
    --recovery_steps 50
```

### Path B (GatedDeltaNet)

```bash
python run_all_4_gated_deltanet_block.py \
    --models distilbert \
    --use_full_attention false \
    --recovery_steps 50
```

## 12.3 Verification Steps

### Verify Weight Transfer

Add logging and check output:

```python
import logging

logging.basicConfig(level=logging.INFO)

# After injection:
for name, mod in model.named_modules():
    if hasattr(mod, "attn") and mod.attn is not None:
        w = mod.attn.in_proj_weight
        print(f"{name}: in_proj_weight norm = {w.norm().item():.4f}")
        # If norm ~= 1e-3, weights are random → silent failure
        # If norm ~= sqrt(3d) * something, weights transferred OK
```

### Verify Causal Masking (GPT-2)

```python
# Compare block output to baseline on first token
x = torch.randn(1, 1, 768)  # single token
y_block = block(x)
# For GPT-2 causal, y_block should equal y_baseline
```

## 12.4 Seeds

```python
seed = 42
torch.manual_seed(seed)
```

## 12.5 Expected Results

| Config | DistilBERT | Status |
|---|---|---|
| Path A (MHA) | 86.5% | 84.5 |
| Path B (DeltaNet) | 50.92% | PASS |

## 12.6 Known Issues

1. **Silent weight transfer failure.** Check norm after injection.
2. **OOM on RoBERTa/GPT-2.** Reduce batch size.
3. **Slow GatedDeltaNet path.** ~90 s per 5 steps.
4. **Non-causal MHA for GPT-2.** Incorrect but not detected by tests.

## 12.7 Data Availability

- Code: `framework/attentions/gated_deltanet_block.py`
- Metrics: `results/unified_gated_deltanet_block/*/summary.csv`
- Logs: `benchmark_logs/gated_deltanet_block_*.log`

---