import os, json
from datetime import datetime
from enum import Enum
from typing import Dict, Any

class ModelCardFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"

def generate_model_card(run, metrics: Dict[str, Any], output_path: str, fmt=ModelCardFormat.MARKDOWN):
    cfg = getattr(run, 'config', {}) or {}
    name = cfg.get('model_name','model')
    task = cfg.get('task_name','task')
    if fmt == ModelCardFormat.MARKDOWN:
        content = _markdown_card(name, task, cfg, metrics)
    elif fmt == ModelCardFormat.HTML:
        content = _html_card(name, task, cfg, metrics)
    else:
        content = json.dumps({"model": name, "task": task, "config": cfg, "metrics": metrics}, indent=2)
        output_path = output_path.replace('.md','.json')
    with open(output_path, 'w') as f: f.write(content)

def _markdown_card(name, task, cfg, metrics):
    return f"""---
language: en
tags: [{task}, transformer]
metrics:
- accuracy: {metrics.get('eval_accuracy','N/A')}
- loss: {metrics.get('eval_loss','N/A')}
---
# {name} – {task.upper()}
## Configuration
- Attention: {cfg.get('attn_type','full')}
- Method: {cfg.get('method','baseline')}
- Epochs: {cfg.get('epochs','?')}
- Batch: {cfg.get('batch_size','?')}
- LR: {cfg.get('learning_rate','?')}
## Performance
- Accuracy: {metrics.get('eval_accuracy','N/A')}
- Loss: {metrics.get('eval_loss','N/A')}
## Artifacts
- Model: model.pth
- Environment: environment.json
- Generated: {datetime.now().isoformat()}
"""

def _html_card(name, task, cfg, metrics):
    return f"<html><body><h1>{name} – {task.upper()}</h1>...</body></html>"