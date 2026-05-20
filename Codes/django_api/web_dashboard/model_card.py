import json
from datetime import datetime

def generate_model_card(run, model_name="", task_name="", final_metrics=None, config=None,
                       license_="mit", language="en", dataset_name="", framework="transformers"):
    metrics = final_metrics or getattr(run, 'final_metrics', {})
    cfg = config or getattr(run, 'config', {})
    model_name = model_name or cfg.get("model_name", "unknown")
    task_name = task_name or cfg.get("task_name", cfg.get("task", "unknown"))

    frontmatter = {
        "language": language,
        "license": license_,
        "tags": [f"task:{task_name}", f"framework:{framework}"],
        "datasets": [dataset_name] if dataset_name else [],
        "metrics": list(metrics.keys()),
    }
    hyperparams_rows = "\n".join(
        f"| {k} | {v} |"
        for k, v in cfg.items()
        if k in ("learning_rate", "batch_size", "epochs", "max_seq_length",
                  "compression_rank", "lora_r", "lora_alpha", "weight_decay",
                  "warmup_steps", "grad_accum", "max_grad_norm")
    )
    metrics_rows = "\n".join(
        f"| {k} | {round(v, 6) if isinstance(v, float) else v} |"
        for k, v in metrics.items()
    )
    return f"""---
{json.dumps(frontmatter, indent=2)}
---

# Model Card for {model_name} on {task_name}

## Hyperparameters
| Parameter | Value |
|-----------|-------|
{hyperparams_rows}

## Results
| Metric | Value |
|--------|-------|
{metrics_rows}

## Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
"""