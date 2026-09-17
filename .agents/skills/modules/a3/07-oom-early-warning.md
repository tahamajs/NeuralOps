# OOM Early Warning — a3

## Purpose

Predict which (backbone, rank, batch_size) triples will OOM before running
them, saving compute.

## Method

Profile at small rank and extrapolate:

```python
def predict_oom_risk(backbone, rank, batch_size, hidden=768, layers=6):
    # Estimate activation memory (forward + backward)
    bytes_per_token = hidden * 4 * 6 * layers  # rough
    total_tokens = batch_size * 128  # seq_len
    activation_mb = total_tokens * bytes_per_token / 1e6
    # Add parameter + optimizer state
    param_mb = 2 * 66e6 * 4 / 1e6
    estimated = activation_mb + param_mb
    return {
        "estimated_mb": estimated,
        "safe": estimated < 6000,  # 6 GB limit
        "risk": "high" if estimated > 6000 else "medium" if estimated > 4000 else "low",
    }
```

## Empirical thresholds for a3

| Backbone | Max safe rank | Max safe batch |
|---|---|---|
| DistilBERT | 384 | 32 |
| RoBERTa | 256 | 16 |
| GPT-2 | 256 | 16 |

Fill from actual runs.

## Output

`papers/modules/a3/metrics/oom_risk.json`
