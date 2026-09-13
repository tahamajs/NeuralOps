#!/usr/bin/env python3
"""
Law 0: Master Benchmark Extraction Script
Scans all real summary.csv files across results/ and produces master_verified_results.json.
No estimated, averaged, or guessed numbers are permitted.
"""

import os
import glob
import csv
import json
from collections import defaultdict

def extract_master_results(project_root):
    results_dir = os.path.join(project_root, 'results')
    summary_files = sorted(glob.glob(os.path.join(results_dir, '**/summary.csv'), recursive=True))
    
    print(f"[LAW 0] Found {len(summary_files)} summary.csv files under {results_dir}")
    
    # Store all runs: (module_name, model_family) -> list of valid runs
    all_runs = defaultdict(list)
    
    for sf in summary_files:
        rel_path = os.path.relpath(sf, project_root)
        parts = rel_path.split(os.sep)
        
        # Infer model family from path if not explicitly in CSV
        inferred_model = None
        for p in parts:
            pl = p.lower()
            if 'distilbert' in pl:
                inferred_model = 'DistilBERT'
            elif 'roberta' in pl:
                inferred_model = 'RoBERTa'
            elif 'gpt2' in pl or 'gpt-2' in pl:
                inferred_model = 'GPT-2'
            elif 'qwen' in pl:
                inferred_model = 'Qwen'
                
        try:
            with open(sf, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mod = row.get('module_name')
                    if not mod or mod.strip().lower() in ('full', 'baseline', 'none', 'module_name', ''):
                        continue
                    
                    mod = mod.strip()
                    
                    # Model family
                    mf = row.get('model_family') or inferred_model or 'Unknown'
                    mfl = mf.lower()
                    if 'distil' in mfl:
                        model_fam = 'DistilBERT'
                    elif 'robert' in mfl:
                        model_fam = 'RoBERTa'
                    elif 'gpt' in mfl:
                        model_fam = 'GPT-2'
                    elif 'qwen' in mfl:
                        model_fam = 'Qwen'
                    else:
                        model_fam = mf.strip()
                        
                    raw_status = (row.get('status') or '').strip().upper()
                    
                    # Accuracy
                    acc_val = None
                    for k in ('accuracy_pct', 'mean_accuracy', 'accuracy'):
                        if row.get(k):
                            try:
                                acc_val = float(row[k])
                                break
                            except (ValueError, TypeError):
                                pass
                                
                    # F1
                    f1_val = None
                    for k in ('f1_score', 'f1_macro', 'f1'):
                        if row.get(k):
                            try:
                                f1_val = float(row[k])
                                break
                            except (ValueError, TypeError):
                                pass
                                
                    # Peak VRAM
                    vram_val = None
                    for k in ('peak_vram_mb', 'vram_mb'):
                        if row.get(k):
                            try:
                                vram_val = float(row[k])
                                break
                            except (ValueError, TypeError):
                                pass
                                
                    # Throughput
                    tps_val = None
                    for k in ('tokens_per_sec', 'throughput_tokens_sec', 'throughput'):
                        if row.get(k):
                            try:
                                tps_val = float(row[k])
                                break
                            except (ValueError, TypeError):
                                pass
                                
                    timestamp = row.get('timestamp') or ''
                    
                    # Normalise status strictly
                    if 'PASS_LOW' in raw_status:
                        status = 'PASS_LOW'
                    elif 'PASS' in raw_status:
                        # Check if accuracy is genuine pass vs 0 / artifact
                        if acc_val is not None and acc_val > 5.0:
                            status = 'PASS'
                        elif acc_val is not None and acc_val <= 5.0:
                            status = 'FAIL'
                        else:
                            status = 'PASS'
                    elif 'FAIL' in raw_status or 'ERROR' in raw_status or 'CRASH' in raw_status:
                        status = 'FAIL'
                    else:
                        status = raw_status if raw_status else 'UNKNOWN'
                        
                    # Completeness metric for canonical selection
                    completeness = (
                        (2 if acc_val is not None and acc_val > 0 else 0) +
                        (3 if f1_val is not None and f1_val > 0 else 0) +
                        (1 if vram_val is not None and vram_val > 0 else 0) +
                        (1 if tps_val is not None and tps_val > 0 else 0) +
                        (2 if 'unified' in rel_path else 0)
                    )
                    
                    all_runs[(mod, model_fam)].append({
                        'module_name': mod,
                        'model_family': model_fam,
                        'status': status,
                        'accuracy_pct': acc_val,
                        'f1_score': f1_val,
                        'peak_vram_mb': vram_val,
                        'throughput_tokens_sec': tps_val,
                        'source_file': rel_path,
                        'timestamp': timestamp,
                        'completeness': completeness
                    })
        except Exception as e:
            pass
            
    print(f"[LAW 0] Extracted records for {len(all_runs)} (module, model) combinations.")
    
    # Pick canonical run per (module, model_family)
    canonical = {}
    for (mod, model_fam), entries in all_runs.items():
        # Sort by completeness desc, then timestamp desc
        sorted_entries = sorted(entries, key=lambda x: (x['completeness'], x['timestamp']), reverse=True)
        canonical[(mod, model_fam)] = sorted_entries[0]
        
    # Build master structured JSON
    master_modules = {}
    for (mod, model_fam), run in canonical.items():
        if mod not in master_modules:
            master_modules[mod] = {
                'module_name': mod,
                'models': {},
                'best_accuracy': 0.0,
                'has_pass': False,
                'has_pass_low': False,
                'is_pure_pass': False,
                'overall_status': 'FAIL'
            }
            
        master_modules[mod]['models'][model_fam] = {
            'status': run['status'],
            'accuracy_pct': run['accuracy_pct'],
            'f1_score': run['f1_score'],
            'peak_vram_mb': run['peak_vram_mb'],
            'throughput_tokens_sec': run['throughput_tokens_sec'],
            'source_file': run['source_file']
        }
        
        acc = run['accuracy_pct'] or 0.0
        if acc > master_modules[mod]['best_accuracy']:
            master_modules[mod]['best_accuracy'] = acc
            
        st = run['status']
        if st == 'PASS' and acc > 5.0:
            master_modules[mod]['has_pass'] = True
        elif st == 'PASS_LOW':
            master_modules[mod]['has_pass_low'] = True

    # Compute overall status
    pass_modules = []
    pass_low_modules = []
    fail_modules = []

    for mod, data in sorted(master_modules.items()):
        if data['has_pass']:
            data['overall_status'] = 'PASS'
            data['is_pure_pass'] = True
            pass_modules.append(mod)
        elif data['has_pass_low']:
            data['overall_status'] = 'PASS_LOW'
            pass_low_modules.append(mod)
        else:
            data['overall_status'] = 'FAIL'
            fail_modules.append(mod)
            
    summary_report = {
        'total_unique_modules': len(master_modules),
        'pass_count': len(pass_modules),
        'pass_low_count': len(pass_low_modules),
        'fail_count': len(fail_modules),
        'pass_modules': pass_modules,
        'pass_low_modules': pass_low_modules,
        'fail_modules': fail_modules
    }
    
    master_data = {
        'metadata': {
            'description': 'Master verified benchmark results extracted strictly from real project summary.csv files under results/',
            'law': 'Law 0 & Law 1 - Antigravity Thesis Standard',
            'summary': summary_report
        },
        'modules': master_modules
    }
    
    out_path_1 = os.path.join(project_root, 'master_verified_results.json')
    out_path_2 = os.path.join(project_root, 'thesis_final', 'master_verified_results.json')
    
    with open(out_path_1, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    with open(out_path_2, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
        
    print(f"[LAW 0] Successfully saved master results to:\n  - {out_path_1}\n  - {out_path_2}")
    return summary_report, master_data

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    summary_report, master_data = extract_master_results(root)
    print("\n" + "="*60)
    print("LAW 0 AUDIT SUMMARY REPORT")
    print("="*60)
    print(f"Total Unique Modules in CSVs: {summary_report['total_unique_modules']}")
    print(f"PASS Modules (at least 1 real PASS > 5% acc): {summary_report['pass_count']}")
    print(f"PASS_LOW Modules (only PASS_LOW, no PASS):   {summary_report['pass_low_count']}")
    print(f"FAIL / Zero / Missing Modules:               {summary_report['fail_count']}")
    print("="*60)
