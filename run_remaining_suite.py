import sys, os, subprocess

modules = [
    ('nexus', 'attention', ['--batch_size', '8', '--grad_accum_steps', '1', '--recovery_steps', '100', '--self_train_steps', '30', '--learning_rate', '0.0001']),
    ('regla_attention', 'attention', ['--batch_size', '8', '--grad_accum_steps', '1', '--recovery_steps', '100', '--self_train_steps', '30', '--learning_rate', '0.0001']),
    ('cosformer', 'attention', ['--batch_size', '4', '--grad_accum_steps', '2', '--recovery_steps', '50', '--self_train_steps', '20', '--learning_rate', '0.0001']),
    ('tri_branch_hybrid', 'attention', ['--batch_size', '4', '--grad_accum_steps', '2', '--recovery_steps', '50', '--self_train_steps', '20', '--learning_rate', '0.0001']),
    ('hla_attention', 'attention', ['--batch_size', '8', '--grad_accum_steps', '1', '--recovery_steps', '50', '--self_train_steps', '20', '--learning_rate', '0.0001']),
    ('hybrid_attention', 'attention', ['--batch_size', '8', '--grad_accum_steps', '1', '--recovery_steps', '50', '--self_train_steps', '20', '--learning_rate', '0.0001']),
]

python_bin = '/mnt/d/venv/bin/python3'
script = 'neuralops/benchmarks/_legacy/kd_quick_test.py'

for mod, mod_type, extra_args in modules:
    print(f'=== STARTING RUN: {mod} ({mod_type}) ===', flush=True)
    cmd = [
        python_bin, script,
        '--modules', mod,
        '--type', mod_type,
        '--model_name', 'gpt2',
        '--rank', '32',
        '--methods', 'C',
        '--use_cache',
        '--device', 'cuda',
    ] + extra_args
    print('Command:', ' '.join(cmd), flush=True)
    res = subprocess.run(cmd, cwd='/mnt/d/SVD_Project')
    print(f'=== FINISHED {mod} with returncode {res.returncode} ===\n', flush=True)
