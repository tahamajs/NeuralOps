import csv
from pathlib import Path

# Load all target modules
modules = [l.strip() for l in open('audit/zeroshot_modules.txt') if l.strip()]

tasks = []
for module in modules:
    # Zero-shot
    zs_done = all((Path(f'results/zeroshot_parallel/{module}/{t}.json')).exists() 
                  for t in ['piqa','arc_easy','arc_challenge','hellaswag','winogrande'])
    if not zs_done:
        tasks.append({'priority': 1, 'task': 'zeroshot', 'module': module, 
                      'dataset': 'zero-shot', 'backbone': 'gpt2'})
    
    # MNLI rank sweep
    if not (Path(f'results/rank_sweep_mnli/{module}/rank_curve.csv')).exists():
        tasks.append({'priority': 2, 'task': 'rank_sweep', 'module': module,
                      'dataset': 'mnli', 'backbone': 'distilbert'})
    
    # AG News rank sweep
    if not (Path(f'results/rank_sweep_agnews/{module}/rank_curve.csv')).exists():
        tasks.append({'priority': 2, 'task': 'rank_sweep', 'module': module,
                      'dataset': 'ag_news', 'backbone': 'distilbert'})
    
    # C4 PPL
    if not (Path(f'results/c4_baselines/{module}/summary.csv')).exists():
        tasks.append({'priority': 3, 'task': 'c4_ppl', 'module': module,
                      'dataset': 'c4', 'backbone': 'gpt2'})

# Sort by priority
tasks.sort(key=lambda x: x['priority'])

with open('audit/master_queue.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['priority','task','module','dataset','backbone'])
    w.writeheader()
    w.writerows(tasks)

print(f'Total tasks: {len(tasks)}')
from collections import Counter
c = Counter(t['task'] for t in tasks)
for k, v in c.items():
    print(f'  {k}: {v}')
