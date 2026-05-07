"""
Auto-generate HuggingFace-compatible Model Cards from run metadata.
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from django.conf import settings


def generate_model_card(
    run: Any,                       # TrainingRun or Job
    model_name: str = "",
    task_name: str = "",
    final_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    license_: str = "mit",
    language: str = "en",
    dataset_name: str = "",
    framework: str = "transformers",
) -> str:
    """
    Generate a Markdown model card following the HuggingFace template.
    Compatible with huggingface_hub ModelCard.push_to_hub().
    """
    metrics = final_metrics or getattr(run, 'final_metrics', {}) or {}
    cfg = config or getattr(run, 'config', {}) or {}
    model_name = model_name or cfg.get("model_name", "unknown")
    task_name = task_name or cfg.get("task_name", cfg.get("task", "unknown"))

    now = datetime.utcnow().isoformat()

    # --- YAML frontmatter ---
    frontmatter = {
        "language": language,
        "license": license_,
        "tags": [
            f"task:{task_name}",
            f"framework:{framework}",
            "generated-by:transformer-toolkit",
        ],
        "datasets": [dataset_name] if dataset_name else [],
        "metrics": list(metrics.keys()),
        "model-index": [{
            "name": f"{model_name}-{task_name}",
            "results": [{
                "task": {"type": task_name},
                "dataset": {"name": dataset_name, "type": task_name},
                "metrics": [{"type": k, "value": v} for k, v in metrics.items()],
            }]
        }] if metrics else [],
    }

    yaml_block = json.dumps(frontmatter, indent=2)

    # --- Markdown body ---
    metrics_rows = "\n".join(
        f"| {k} | {round(v, 6) if isinstance(v, float) else v} |"
        for k, v in metrics.items()
    )

    hyperparams_rows = "\n".join(
        f"| {k} | {v} |"
        for k, v in cfg.items()
        if k in ("learning_rate", "batch_size", "epochs", "max_seq_length",
                  "compression_rank", "lora_r", "lora_alpha", "weight_decay",
                  "warmup_steps", "grad_accum", "max_grad_norm")
    )

    card = f"""---
{yaml_block}
---

# Model Card for {model_name} fine-tuned on {task_name}

## Model Details

- **Base Model**: [{model_name}](https://huggingface.co/{model_name})
- **Task**: {task_name}
- **Framework**: {framework}
- **Generated**: {now}

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
{hyperparams_rows}

## Evaluation Results

| Metric | Value |
|--------|-------|
{metrics_rows}

## Intended Use

This model was fine-tuned on the **{task_name}** task using the Transformer Toolkit platform.

## Limitations & Bias

Automated evaluation may not capture all edge cases. Consider further testing before deploying.

## How to Use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")