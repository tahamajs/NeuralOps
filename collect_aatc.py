import os
import json
import csv
import glob
import pandas as pd

aatc_files = glob.glob('results/**/*aatc_xkv*.json', recursive=True) + glob.glob('results/**/*aatc_xkv*.csv', recursive=True)

all_metrics = []

def flatten(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res.update(flatten(v))
        else:
            res[k] = v
    return res

for f in aatc_files:
    if f.endswith('.json') and 'metrics' in f:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    for d in data:
                        d = flatten(d)
                        d['source_file'] = f
                        all_metrics.append(d)
                elif isinstance(data, dict):
                    data = flatten(data)
                    data['source_file'] = f
                    all_metrics.append(data)
        except:
            pass

master_csvs = glob.glob('results/*summary.csv') + glob.glob('results/*table*.csv')
for mcsv in master_csvs:
    try:
        df = pd.read_csv(mcsv)
        aatc_rows = df[df['module_name'].astype(str).str.contains('aatc', case=False, na=False)]
        for _, row in aatc_rows.iterrows():
            row_dict = row.to_dict()
            row_dict['source_file'] = mcsv
            all_metrics.append(row_dict)
    except Exception as e:
        pass

grouped = {}
for m in all_metrics:
    file_path = m.get('source_file', '')
    
    acc = m.get('eval_accuracy', m.get('accuracy_pct', m.get('accuracy', None)))
    if acc is None: acc = 'N/A'
    if isinstance(acc, float) and acc < 1.0: acc = round(acc * 100, 2)
    
    comp_rate = m.get('compression_rate', m.get('compression_ratio', m.get('compression_ratio_pct', 'N/A')))
        
    vram = m.get('peak_vram_mb', m.get('Peak VRAM (MB)', 'N/A'))
    tps = m.get('throughput_tokens_sec', m.get('tokens_per_sec', 'N/A'))
    
    rank = 'auto'
    if 'r16' in file_path: rank = 16
    elif 'r32' in file_path: rank = 32
    elif 'r64' in file_path: rank = 64
    elif 'r128' in file_path: rank = 128
    elif 'r256' in file_path: rank = 256
    elif 'r384' in file_path: rank = 384
    
    if 'gpt2' in file_path.lower(): task = "GPT-2 (WikiText-2 Causal LM)"
    elif 'sst2' in file_path.lower() or 'distilbert' in file_path.lower() or 'bert' in file_path.lower(): task = "DistilBERT (SST-2 Classification)"
    elif 'agnews' in file_path.lower(): task = "AG News"
    elif 'zeroshot' in file_path.lower(): task = "Zero-Shot"
    else: task = "Other"
        
    ppl = m.get('eval_perplexity', m.get('perplexity', 'N/A'))
        
    key = f"{task} - Rank: {rank}"
    if key not in grouped: grouped[key] = []
        
    entry = {
        'Accuracy': f"{acc}%" if acc != 'N/A' else 'N/A',
        'Perplexity': round(ppl, 2) if isinstance(ppl, float) else ppl,
        'Compression Rate': comp_rate,
        'Peak VRAM (MB)': round(vram, 2) if isinstance(vram, float) else vram,
        'Tokens/sec': round(tps, 2) if isinstance(tps, float) else tps,
        'File': file_path
    }
    
    if not any(e['File'] == entry['File'] for e in grouped[key]):
        grouped[key].append(entry)

out_lines = ["# Complete AATC-XKV Benchmark Results\n"]
out_lines.append("> **Module Description**: Attention-Aware Cross-Layer KV Cache Compression (`aatc_xkv_compressor`). Extremely efficient memory footprint reduction for caching keys and values.\n")

for k, v in grouped.items():
    out_lines.append(f"## {k}")
    for item in v:
        out_lines.append(f"- **Accuracy**: {item['Accuracy']} | **PPL**: {item['Perplexity']} | **Comp. Rate**: {item['Compression Rate']} | **VRAM**: {item['Peak VRAM (MB)']} MB | **TPS**: {item['Tokens/sec']}")
        out_lines.append(f"  - *Source: `{item['File']}`*")
    out_lines.append("")

with open('/Users/tahamajs/.gemini/antigravity-cli/brain/6841c82f-672b-40a2-8ffe-2c555bdd27ed/AATC_RESULTS_DUMP.md', 'w') as f:
    f.write("\n".join(out_lines))
