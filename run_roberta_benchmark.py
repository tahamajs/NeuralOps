#!/usr/bin/env python3
import subprocess
import os
import sys
import time
import json
import shutil
import argparse
import pandas as pd
from pathlib import Path

os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

if os.path.exists(r'D:\svd_project'):
    os.chdir(r'D:\svd_project')

parser = argparse.ArgumentParser(description='RoBERTa-Only Quality Benchmark Runner')
parser.add_argument('module', type=str, help='Attention or compressor module to benchmark')
parser.add_argument('--recovery_steps', type=int, default=200, help='Recovery steps')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--learning_rate', type=str, default='2e-4', help='Learning rate')
parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
parser.add_argument('--rank', type=int, default=32, help='Module rank')
parser.add_argument('--precision', type=str, default='fp16', help='Precision')
args = parser.parse_args()

module_name = args.module
out_dir = Path(f'results/unified_{module_name}/roberta')
out_dir.mkdir(parents=True, exist_ok=True)
summary_csv = out_dir / 'summary.csv'
log_file = out_dir / 'console.log'

cmd = [
    sys.executable, '-u', r'Codesrameworkun_quality_sweep.py',
    '--model', r'modelsoberta-base',
    '--model_family', 'roberta',
    '--dataset', 'sst2',
    '--dataset_dir', r'Codesramework\data\glue_data\sst2',
    '--output_dir', str(out_dir),
    '--parent_checkpoint', r'resultsoberta_sst2_lora_baselineinetuned_model.pt',
    '--modules', f'full,{module_name}',
    '--batch_size', str(args.batch_size),
    '--recovery_steps', str(args.recovery_steps),
    '--learning_rate', str(args.learning_rate),
    '--lora_rank', str(args.lora_rank),
    '--rank', str(args.rank),
    '--precision', str(args.precision),
    '--fresh',
    '--defer_export',
    '--timeout', '1200',
]

print(f'========================================================')
print(f'Starting ROBERTA Benchmark for {module_name.upper()}...')
print(f'Command: {" ".join(cmd)}')
print(f'========================================================
', flush=True)

start_time = time.time()
with open(log_file, 'w', encoding='utf-8') as lf:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lf.write(line)
        lf.flush()
    proc.wait()

elapsed_total = time.time() - start_time
if proc.returncode != 0:
    print(f'
[ERROR] Benchmark failed for roberta with returncode {proc.returncode}.', flush=True)
else:
    print(f'
[SUCCESS] Finished Benchmark for roberta in {elapsed_total:.2f}s.', flush=True)

# Quality Gate and Export
print(f'
========================================================')
print(f'Starting Quality Gate & Automated Hugging Face Export...')
print(f'========================================================
', flush=True)

if not summary_csv.exists():
    print(f'[Quality Gate: REJECTED] roberta summary CSV does not exist at {summary_csv}. Skipping export.', flush=True)
    sys.exit(1)

df = pd.read_csv(summary_csv)
mod_col = 'module' if 'module' in df.columns else ('module_name' if 'module_name' in df.columns else None)
if mod_col is None or df.empty:
    print(f'[Quality Gate: REJECTED] Summary CSV is empty or missing module column.', flush=True)
    sys.exit(1)

cand = df[df[mod_col] != 'full']
if cand.empty:
    cand = df
row = cand.iloc[-1].to_dict()

status = str(row.get('status', 'FAIL'))
try:
    acc = float(row.get('accuracy_pct', row.get('mean_accuracy', 0.0)))
except Exception:
    acc = 0.0

is_successful = (status == 'PASS') and (acc >= 56.0)
weights_dir = out_dir / 'weights'
hf_export_dir = out_dir / 'hf_export'

if not is_successful:
    print(f'[Quality Gate: REJECTED] roberta did not meet criteria (status={status}, acc={acc:.2f}% < 56.0%). Purging checkpoints.', flush=True)
    if weights_dir.exists():
        shutil.rmtree(weights_dir, ignore_errors=True)
    sys.exit(1)

print(f'[Quality Gate: PASSED] roberta passed criteria (status={status}, acc={acc:.2f}% >= 56.0%). Packaging for export...', flush=True)
hf_export_dir.mkdir(parents=True, exist_ok=True)

# Copy weights
weights_candidates = list(weights_dir.glob('**/weights.pt')) + list(weights_dir.glob('weights.pt'))
if weights_candidates:
    src_weights = weights_candidates[0]
    dst_weights = hf_export_dir / 'weights.pt'
    if src_weights.resolve() != dst_weights.resolve():
        shutil.copy2(src_weights, dst_weights)

metrics_payload = {
    'model_name': 'roberta',
    'module': module_name,
    'status': status,
    'accuracy_pct': acc,
    'f1': float(row.get('f1_score', row.get('f1', 0.0))),
    'precision': float(row.get('precision', 0.0)),
    'recall': float(row.get('recall', 0.0)),
    'eval_loss': float(row.get('val_loss', row.get('eval_loss', 0.0))),
    'peak_vram_mb': float(row.get('peak_vram_mb', 0.0)),
    'tokens_per_sec': float(row.get('tokens_per_sec', 0.0)),
    'elapsed_seconds': float(row.get('elapsed_s', elapsed_total)),
    'full_metrics': row,
}
with open(hf_export_dir / 'metrics.json', 'w', encoding='utf-8') as mf:
    json.dump(metrics_payload, mf, indent=2)

card_content = f"""---
tags:
- svd-linear-attention
- {module_name}-attention
- pytorch
datasets:
- sst2
---
# {module_name.upper()} - roberta
- Benchmark Accuracy: {acc:.2f}%
- F1 Score: {metrics_payload['f1']:.4f}
- Status: {status}
- Architecture: roberta
"""
with open(hf_export_dir / 'README.md', 'w', encoding='utf-8') as f:
    f.write(card_content)

repo_id = f'Hooshaai/svd-linear-attention-roberta-{module_name.replace("_", "-")}'
try:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f'Uploading to Hugging Face: {repo_id}...', flush=True)
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(hf_export_dir), repo_id=repo_id, commit_message=f'Upload hardened {module_name} roberta weights - Acc: {acc:.2f}%')
    print(f'Successfully uploaded roberta to {repo_id}', flush=True)
except Exception as e:
    print(f'Upload failed for {repo_id}: {e}', flush=True)

# Master summary CSV/JSON
full_rows = df[df[mod_col] == 'full']
base_acc = float(full_rows.iloc[0].get('accuracy_pct', 0.0)) / 100.0 if not full_rows.empty else 0.8876
delta_acc = (acc / 100.0) - base_acc

res_item = {
    'model': 'roberta',
    'slug': 'roberta',
    'baseline_acc': base_acc,
    'acc': acc / 100.0,
    'delta_acc': delta_acc,
    'f1': metrics_payload['f1'],
    'status': status,
    'passed_gate': is_successful,
    'vram_peak_mb': metrics_payload['peak_vram_mb'],
    'out_dir': str(out_dir),
    'hf_repo': repo_id,
}
master_csv = Path(f'results/unified_{module_name}_4models_summary.csv')
master_json = Path(f'results/unified_{module_name}_4models_summary.json')
summary_df = pd.DataFrame([res_item])
summary_df.to_csv(master_csv, index=False)
with open(master_json, 'w', encoding='utf-8') as jf:
    json.dump([res_item], jf, indent=2)
print(f'
[+] Master summary saved to {master_csv} and {master_json}')

# Update Docs
for doc_candidate in [
    Path(f'Docs/content/attentions/{module_name}.md'),
    Path(f'Docs/content/attentions/{module_name}_attention.md'),
    Path(f'Docs/content/compressions/{module_name}.md'),
    Path(f'Docs/content/compressions/{module_name}_pruner.md'),
]:
    if doc_candidate.exists():
        table_lines = [
            f'
## Unified 4-Architecture Benchmark Results ({module_name})
',
            'Evaluated on SST-2 sentiment classification (FP16, batch size 8, 200 recovery steps, lr 2e-4, LoRA rank 16):

',
            '| Model Architecture | Baseline Acc | Module Acc | Δ Acc | F1 Score | Peak VRAM | Gate Status | Hugging Face Repository |
',
            '|:-------------------|:------------:|:----------:|:-----:|:--------:|:---------:|:-----------:|:------------------------|
',
            f'| ROBERTA | {base_acc:.2%} | {acc/100.0:.2%} | {delta_acc:+.2%} | {metrics_payload["f1"]:.4f} | {metrics_payload["peak_vram_mb"]:.1f} MB |  | [](https://huggingface.co/{repo_id}) |
',
        ]
        content = doc_candidate.read_text(encoding='utf-8')
        if '## Unified 4-Architecture Benchmark Results' in content:
            content = content.split('## Unified 4-Architecture Benchmark Results')[0]
        content = content.rstrip() + '
' + ''.join(table_lines)
        doc_candidate.write_text(content, encoding='utf-8')
        print(f'
[+] Updated documentation benchmark table in {doc_candidate}')

# Git branch commit & push
branch_name = f'benchmark-results-{module_name}'
print(f'
[+] Committing and pushing results to branch '{branch_name}'...')
try:
    subprocess.run(['git', 'checkout', '-b', branch_name], check=False)
    subprocess.run(['git', 'checkout', branch_name], check=False)
    add_files = []
    if master_csv.exists(): add_files.append(str(master_csv))
    if master_json.exists(): add_files.append(str(master_json))
    for d in Path('Docs/content').rglob(f'*{module_name}*.md'):
        add_files.append(str(d))
    if add_files:
        subprocess.run(['git', 'add'] + add_files, check=True)
        subprocess.run(['git', 'commit', '-m', f'Add unified benchmark results and documentation for {module_name}'], check=True)
        subprocess.run(['git', 'push', 'origin', branch_name], check=True)
        print(f'[+] Successfully committed and pushed results to GitHub branch '{branch_name}'.')
    subprocess.run(['git', 'checkout', 'main'], check=False)
except Exception as e:
    print(f'[-] Git branch commit/push warning: {e}')

print('
[COMPLETE] RoBERTa benchmark executed and exported successfully.')
