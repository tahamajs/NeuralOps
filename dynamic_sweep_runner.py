#!/usr/bin/env python3
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import psutil
except ImportError as exc:
    raise SystemExit('psutil is required') from exc

ROOT = Path('/mnt/d/SVD_Project')
PYTHON = '/mnt/d/venv/bin/python3'
BENCH = ROOT / 'scripts/benchmarks/benchmark_all_108_cuda_isolated.py'
OUTDIR = ROOT / 'results/benchmarks_v1.4.0'
MODULES = ['entropy_pruning', 'flash_gqa', 'sparse_attention', 'spectral_mamba', 'split_head_norm', 'swiglu', 'tt_contraction', 'varlen_attention']
TIMEOUT = 180

def resources_ok():
    if psutil.virtual_memory().available < 4 * 1024**3:
        return False
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
        return any(float(x.strip()) >= 2000 for x in out.splitlines() if x.strip())
    except Exception:
        return False

def run_one(module):
    while not resources_ok():
        time.sleep(15)
    output = Path(f'/mnt/d/sweep_dynamic_{module}.json')
    env = os.environ.copy()
    env.update(TMPDIR='/mnt/d/tmp', TORCH_EXTENSIONS_DIR='/mnt/d/.torch_extensions', TRITON_CACHE_DIR='/mnt/d/.triton_cache', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MAX_JOBS='1')
    cmd = [PYTHON, str(BENCH), '--modules', module, '--output', str(output)]
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {'module': module, 'status': 'TIMEOUT', 'reason': f'exceeded {TIMEOUT}s'}
    if output.exists():
        try:
            data = json.loads(output.read_text())
            if isinstance(data, list) and data:
                return data[0]
        except Exception as exc:
            return {'module': module, 'status': 'ERROR', 'reason': str(exc)}
    return {'module': module, 'status': 'NO_RESULT'}

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, module) for module in MODULES]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    master = OUTDIR / 'master_108_FULL.json'
    try:
        base = json.loads(master.read_text())
    except Exception:
        base = []
    by_module = {row.get('module'): row for row in base if isinstance(row, dict) and row.get('module')}
    for row in results:
        by_module[row['module']] = row
    master.write_text(json.dumps(list(by_module.values()), indent=2) + '\n')
    print(f'MERGED_ROWS={len(by_module)}', flush=True)

if __name__ == '__main__':
    main()
