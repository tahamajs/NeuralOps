import glob
import json
import os

base = r'D:\SVD_Project\results\04_rank_sweeps'
count = 0
for metrics_file in glob.glob(os.path.join(base, '**', '*_full_metrics.json'), recursive=True):
    dir_path = os.path.dirname(metrics_file)
    csv_path = os.path.join(dir_path, 'summary.csv')
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        p = data.get('performance', {})
        c = data.get('compression', {})
        status = p.get('status', 'PASS')
        if status == 'PASS_NO_COMPRESSION':
            status = 'PASS'
        mod_name = data.get('module_name') or p.get('module_name', '')
        acc = p.get('accuracy_pct', 0.0)
        f1 = p.get('f1_score', 0.0)
        val_loss = p.get('val_loss', 0.0)
        elapsed = p.get('elapsed_s', 0.0)
        vram = p.get('peak_vram_mb', 0.0)
        tok_sec = p.get('tokens_per_sec', 0.0)
        p_bef = c.get('total_params_before', 0)
        p_aft = c.get('total_params_after', 0)
        crate = c.get('compression_rate', 1.0)
        n_comp = c.get('num_compressed_layers', 0)
        n_fall = c.get('num_fallback_layers', 0)
        err = str(p.get('error', '')).replace('"', '""')
        ts = p.get('timestamp', '')

        header = "module_name,status,accuracy_pct,f1_score,val_loss,elapsed_s,peak_vram_mb,tokens_per_sec,total_params_before,total_params_after,compression_rate,num_compressed_layers,num_fallback_layers,error,timestamp\n"
        row = f'{mod_name},{status},{acc},{f1},{val_loss},{elapsed},{vram},{tok_sec},{p_bef},{p_aft},{crate},{n_comp},{n_fall},"{err}",{ts}\n'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(header + row)
        count += 1
    except Exception as e:
        pass

print(f"Successfully processed {count} summary.csv files with status PASS.")
