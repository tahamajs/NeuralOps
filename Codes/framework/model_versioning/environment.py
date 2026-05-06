import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, Any

def get_git_info() -> Dict[str, Any]:
    """Return Git commit, branch, dirty status, and remote URL."""
    try:
        from git import Repo
        repo = Repo('.', search_parent_directories=True)
        return {
            'commit_hash': repo.head.commit.hexsha,
            'branch': repo.active_branch.name,
            'dirty': repo.is_dirty(untracked_files=True),
            'repository': repo.remotes.origin.url if repo.remotes else None,
        }
    except Exception as e:
        return {'error': str(e)}

def get_dvc_data_commit() -> str:
    """Return the latest Git commit that touched any .dvc file."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H', '--', '*.dvc'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception:
        return None

def get_pip_freeze() -> str:
    """Return pip freeze output as a string."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True
        )
        return result.stdout
    except Exception:
        return ""

def get_conda_env() -> str:
    """Return conda environment export if available."""
    try:
        result = subprocess.run(
            ['conda', 'env', 'export'],
            capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""

def capture_environment(output_dir: str) -> Dict[str, Any]:
    """Capture full environment snapshot to a JSON file and return it."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "git": get_git_info(),
        "dvc_commit": get_dvc_data_commit(),
        "pip_freeze": get_pip_freeze(),
    }
    conda = get_conda_env()
    if conda:
        env["conda_env"] = conda

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "environment.json"), 'w') as f:
        json.dump(env, f, indent=2)
    return env

def generate_model_card(run, metrics: Dict[str, Any], output_path: str):
    """Generate a Markdown model card using the run's config and metrics."""
    config = getattr(run, 'config', {}) or {}
    name = config.get('model_name', 'model')
    task = config.get('task_name', 'task')
    card = f"""---
language: en
tags:
- {task}
- transformer
metrics:
- name: accuracy
  value: {metrics.get('eval_accuracy', 'N/A')}
- name: loss
  value: {metrics.get('eval_loss', 'N/A')}
---

# {name} – {task.upper()}

## Training Configuration
- **Model**: {name}
- **Attention**: {config.get('attn_type', 'full')}
- **Method**: {config.get('method', 'baseline')}
- **Epochs**: {config.get('epochs', '?')}
- **Batch Size**: {config.get('batch_size', '?')}
- **Learning Rate**: {config.get('learning_rate', '?')}

## Performance
- **Accuracy**: {metrics.get('eval_accuracy', 'N/A')}
- **Loss**: {metrics.get('eval_loss', 'N/A')}

## Artifacts
- Model file: `model.pth`
- Environment: `environment.json`
- Generated: {datetime.now().isoformat()}
"""
    with open(output_path, 'w') as f:
        f.write(card)