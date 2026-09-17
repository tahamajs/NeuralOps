# Spectral Analysis

## Purpose

Characterize how compression affects the weight matrix spectrum.

## Per (backbone, rank, layer)

- SVD spectrum: `metrics/spectra/<layer>.json`
- Effective rank: number of singular values with energy > 1% of max
- Stable rank: `‖W‖_F² / ‖W‖_2²`
- Nuclear norm: `Σ σ_i`
- Condition number: `σ_max / σ_min`
- Frobenius norm before/after
- Weight drift: `‖W_orig - W_comp‖_F / ‖W_orig‖_F`
- Sparsity: fraction of zeros after compression

## Implementation

```python
import torch

def analyze_layer(W_orig, W_comp):
    U_o, S_o, _ = torch.linalg.svd(W_orig.float(), full_matrices=False)
    U_c, S_c, _ = torch.linalg.svd(W_comp.float(), full_matrices=False)
    fro_o = torch.linalg.norm(W_orig.float()).item()
    fro_c = torch.linalg.norm(W_comp.float()).item()
    drift = torch.linalg.norm(W_orig.float() - W_comp.float()).item() / (fro_o + 1e-12)
    energy = (S_o ** 2).cumsum(0) / (S_o ** 2).sum()
    eff_rank = int((energy < 0.99).sum().item()) + 1
    stable_rank = (fro_o ** 2) / (S_o[0] ** 2 + 1e-12)
    cond = (S_o[0] / (S_o[-1] + 1e-12)).item()
    return {
        "frobenius_orig": fro_o, "frobenius_comp": fro_c,
        "weight_drift": drift, "effective_rank": eff_rank,
        "stable_rank": stable_rank.item(), "condition_number": cond,
        "spectrum_orig": S_o.tolist(), "spectrum_comp": S_c.tolist(),
        "sparsity_comp": float((W_comp == 0).float().mean().item()),
    }
```

## Output

- `papers/modules/<module>/metrics/spectra/<layer>.json`
- `papers/modules/<module>/metrics/effective_rank.csv`
- `papers/modules/<module>/metrics/stable_rank.csv`
- `papers/modules/<module>/metrics/frobenius.csv`
- `papers/modules/<module>/metrics/nuclear_norm.csv`
- `papers/modules/<module>/metrics/condition_number.csv`
- `papers/modules/<module>/metrics/weight_drift.csv`
- `papers/modules/<module>/metrics/sparsity.csv`
- `papers/modules/<module>/figures/weight_histograms.png`
- `papers/modules/<module>/figures/per_layer_error.png`
