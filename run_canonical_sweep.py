import sys, os
from pathlib import Path
sys.path.insert(0, r'D:\SVD_Project\Codes')

from framework.run_quality_sweep import run_quality_benchmark_sweep

# Read canonical modules list
canon_file = Path(r'D:\SVD_Project\canonical_modules.txt')
with open(canon_file, 'r', encoding='utf-8') as f:
    modules = [
        line.split('#')[0].strip()
        for line in f
        if line.strip() and not line.strip().startswith('#')
    ]

print(f"Loaded {len(modules)} canonical modules from {canon_file}")

run_quality_benchmark_sweep(
    modules=modules,
    models=['distilbert'],
    datasets=['sst2'],
    seeds=[42],
    epochs=1,
    batch_size=64,
    timeout=600,
    precision='fp16',
    output_dir='results/all_modules_paper_run',
    pretrained=True
)
