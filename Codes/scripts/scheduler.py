import subprocess
import itertools
import json
from datetime import datetime

def run_grid_search(config_template, model_names, ranks, tasks, attention_types=['svd']):
    """Run all combinations of hyperparameters."""
    combinations = list(itertools.product(model_names, ranks, tasks, attention_types))
    results = []
    for model, rank, task, attn in combinations:
        run_name = f"{model}_{attn}_{rank}_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cmd = [
            'python', 'experiments/run_experiment.py',
            '--model', model,
            '--attention', attn,
            '--rank', str(rank),
            '--task', task,
            '--run_name', run_name
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        results.append({
            'model': model,
            'rank': rank,
            'task': task,
            'attention': attn,
            'run_name': run_name,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
        with open('scheduler_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    return results