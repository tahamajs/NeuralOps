import subprocess, sys, time

try:
    import run_all_modules_auto
    modules = ','.join(run_all_modules_auto.WHITELIST_MODULES)
except Exception:
    modules = None

datasets = [
    ('cola', 'models/distilbert_cola', 'Codes/framework/data/glue_data/cola'),
    ('mrpc', 'models/distilbert_mrpc', 'Codes/framework/data/glue_data/mrpc'),
    ('qnli', 'models/distilbert_qnli', 'Codes/framework/data/glue_data/qnli'),
    ('qqp',  'models/distilbert_qqp',  'Codes/framework/data/glue_data/qqp'),
    ('rte',  'models/distilbert_rte',  'Codes/framework/data/glue_data/rte')
]

print("Starting global dataset queue...")

for ds, model, data_dir in datasets:
    print(f"\n=============================================")
    print(f"Queueing dataset: {ds}")
    print(f"=============================================")
    cmd = [
        sys.executable, 'scripts/run_glue_parallel.py',
        '--dataset', ds,
        '--model', model,
        '--dataset_dir', data_dir,
        '--parallel', '1',
        '--recovery_steps', '500'
    ]
    if modules:
        cmd.extend(['--modules', modules])
    
    subprocess.run(cmd)

print('All GLUE datasets completed!')
