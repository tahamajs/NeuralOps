import subprocess, sys
modules = ['tpa', 'gated_delta_net', 'weighted_ensemble']
ranks = [128]
for mod in modules:
    for rank in ranks:
        cmd = [sys.executable, '-u', f'Codes/runners/run_all_4_{mod}.py', '--models', 'distilbert', '--recovery_steps', '10', '--rank', str(rank)]
        print(f'Running: {mod} {rank}')
        subprocess.run(cmd, check=False)
