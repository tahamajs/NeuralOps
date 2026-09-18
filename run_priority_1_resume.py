import os, sys, subprocess
from pathlib import Path

modules = ['phantom', 'jamba', 'unified_rope']
model_path = r'D:/SVD_Project/models/gpt2'
output_base = r'D:/SVD_Project/results/wikitext2_ppl'

for idx, m in enumerate(modules, 1):
    print(f'=== [{idx}/{len(modules)}] Evaluating {m} ===', flush=True)
    out_dir = f'{output_base}/{m}'
    cmd = [
        sys.executable,
        r'D:/SVD_Project/Codes/framework/eval_perplexity.py',
        '--model', model_path,
        '--module', m,
        '--output_dir', out_dir
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(res.stdout, flush=True)
        if res.returncode != 0:
            print(f'ERROR in {m}:', res.stderr[-500:], flush=True)
    except Exception as e:
        print(f'FAILED {m}: {e}', flush=True)

print('Priority 1 Resumed PPL modules finished!', flush=True)
