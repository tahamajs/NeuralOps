#!/bin/bash
while true; do
    ssh gpu "echo Connection OK && cd /d D:\SVD_Project && python -c \"
import glob, json
modules = ['aegis_hybrid', 'svd_compressed', 'tpa', 'gated_delta_net', 'weighted_ensemble']
for mod in modules:
    files = glob.glob(f'results/unified_{mod}/distilbert/*_{mod}_full_metrics.json')
    if files:
        with open(files[-1]) as f:
            data = json.load(f)
            print(f'{mod}: {data.get(\\\"status\\\", \\\"UNKNOWN\\\")} {data.get(\\\"accuracy_pct\\\", 0):.2f}%')
    else:
        print(f'{mod}: NOT FOUND')
\""
    if [ $? -eq 0 ]; then
        break
    else
        sleep 5
    fi
done
