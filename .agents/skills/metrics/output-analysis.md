# Output Analysis

## Purpose

Measure how compression changes model outputs, not just accuracy.

## Metrics

- Prediction agreement with baseline: fraction of samples where argmax matches
- Logit correlation: Pearson and Spearman
- KL divergence per sample: `D_KL(p_orig ‖ p_comp)`
- Expected Calibration Error (ECE)
- Reliability diagram
- ROC-AUC per class
- Per-class precision/recall/F1
- Top-5 disagreement samples

## Implementation

```python
import torch, numpy as np
from scipy.stats import pearsonr, spearmanr

def output_analysis(model_orig, model_comp, eval_loader, device):
    agree = 0
    kl_divs = []
    logits_o_all = []
    logits_c_all = []
    for batch in eval_loader:
        with torch.no_grad():
            out_o = model_orig(**batch).logits
            out_c = model_comp(**batch).logits
        pred_o = out_o.argmax(-1)
        pred_c = out_c.argmax(-1)
        agree += (pred_o == pred_c).sum().item()
        p_o = torch.softmax(out_o.float(), -1)
        p_c = torch.softmax(out_c.float(), -1)
        kl = (p_o * (p_o.log() - p_c.log())).sum(-1)
        kl_divs.extend(kl.cpu().tolist())
        logits_o_all.append(out_o.cpu())
        logits_c_all.append(out_c.cpu())
    ...
```

## Output

- `papers/modules/<module>/metrics/pred_agreement.json`
- `papers/modules/<module>/metrics/logit_correlation.json`
- `papers/modules/<module>/metrics/output_kl.jsonl`
- `papers/modules/<module>/metrics/ece.json`
- `papers/modules/<module>/metrics/roc_auc.csv`
- `papers/modules/<module>/metrics/per_class_metrics.csv`
- `papers/modules/<module>/metrics/confusion_matrices.npz`
- `papers/modules/<module>/figures/reliability_diagram.png`
- `papers/modules/<module>/figures/pr_curves.png`
- `papers/modules/<module>/examples/disagreements.json`
