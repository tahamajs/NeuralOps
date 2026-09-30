import subprocess, sys  
import run_all_modules_auto  
modules = ','.join(run_all_modules_auto.WHITELIST_MODULES)  
datasets = [('sst2', 'models/distilbert_sst2', 'Codes/framework/data/glue_data/sst2')]  
for ds, model, data_dir in datasets:  
    print(f'Queueing {ds}')  
    subprocess.run([sys.executable, 'scripts/run_glue_parallel.py', '--dataset', ds, '--model', model, '--dataset_dir', data_dir, '--parallel', '2', '--recovery_steps', '500', '--modules', modules])  
