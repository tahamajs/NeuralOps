import sys, os, subprocess

modules = ['nexus', 'regla_attention', 'hla_attention', 'simple_ssm']

cmd = [
    '/mnt/d/venv/bin/python3', '-m', 'neuralops.benchmarks._legacy.run_quality_sweep',
    '--model', 'gpt2',
    '--modules', *modules,
    '--finetune_steps', '50',
    '--finetune_mode', 'lora',
    '--batch_size', '4',
    '--seq_len', '256',
    '--output_dir', 'results/quality_sweep_verified',
    '--fresh',
]

print('Launching Quality Sweep with Training:')
print(' '.join(cmd), flush=True)

res = subprocess.run(cmd, cwd='/mnt/d/SVD_Project')
print(f'Quality sweep finished with exit code {res.returncode}')
