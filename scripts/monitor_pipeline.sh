#!/bin/bash
while true; do
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "===== Pipeline Status @ $TS ====="
    
    ssh gpu "powershell -Command \"Get-Content D:\\SVD_Project\\master_pipeline.log -Tail 20\""
    
    echo "--- GPU ---"
    ssh gpu "nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader"
    
    echo "--- Completed counts ---"
    ssh gpu "cd /d D:\SVD_Project && python -c \"
from pathlib import Path
zs = [d for d in Path('results/zeroshot_parallel').glob('*/') if all((d/f'{t}.json').exists() for t in ['piqa','arc_easy','arc_challenge','hellaswag','winogrande'])]
mnli = list(Path('results/rank_sweep_mnli').glob('*/'))
ag = list(Path('results/rank_sweep_agnews').glob('*/'))
c4 = list(Path('results/c4_baselines').glob('*/'))
print(f'zeroshot: {len(zs)}/183')
print(f'mnli:     {len(mnli)}/183')
print(f'agnews:   {len(ag)}/183')
print(f'c4:       {len(c4)}/183')
\""
    
    echo ""
    sleep 1800  # 30 min
done
