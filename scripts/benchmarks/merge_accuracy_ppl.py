import json
import csv
import os

p0_file = 'results/accuracy_and_ppl_benchmark_taha_part0.json'
p1_file = 'results/accuracy_and_ppl_benchmark_tahoora_part1.json'

with open(p0_file, 'r') as f:
    d0 = json.load(f)
with open(p1_file, 'r') as f:
    d1 = json.load(f)

res0 = d0.get('results', [])
res1 = d1.get('results', [])

all_res = {r['module']: r for r in (res0 + res1)}
sorted_modules = sorted(all_res.keys())
sorted_results = [all_res[m] for m in sorted_modules]

base_acc = d0.get('baseline_roberta_accuracy_pct', 95.0)
base_ppl = d0.get('baseline_gpt2_ppl', 27.56)

merged = {
    'baseline_roberta_accuracy_pct': base_acc,
    'baseline_gpt2_ppl': base_ppl,
    'total_modules': len(sorted_results),
    'results': sorted_results
}

os.makedirs('results', exist_ok=True)

with open('results/accuracy_and_ppl_benchmark_all131.json', 'w') as f:
    json.dump(merged, f, indent=2)

with open('results/accuracy_and_ppl_benchmark_all131.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['module', 'accuracy_pct', 'perplexity', 'duration_s', 'error'])
    writer.writerow(['BASELINE', f'{base_acc:.2f}', f'{base_ppl:.2f}', '0.0', ''])
    for r in sorted_results:
        acc_val = f"{r['accuracy_pct']:.2f}" if r.get('accuracy_pct') is not None else ''
        ppl_val = f"{r['perplexity']:.2f}" if r.get('perplexity') is not None else ''
        writer.writerow([
            r['module'],
            acc_val,
            ppl_val,
            f"{r['duration_s']:.2f}",
            r.get('error', '')
        ])

lines = []
lines.append('# Attention Modules Benchmark: Zero-Shot Accuracy & Perplexity (All 131 Modules)\n')
lines.append('- **Hardware Split**: RTX 4070 (modules 1-66) + GTX 1060 (modules 67-131)')
lines.append(f'- **RoBERTa SST-2 Baseline Accuracy**: {base_acc:.2f}%')
lines.append(f'- **GPT-2 WikiText-2 Baseline PPL**: {base_ppl:.2f}')
lines.append(f'- **Total Modules Evaluated**: {len(sorted_results)}\n')
lines.append('| # | Module | RoBERTa SST-2 Acc (%) | GPT-2 WikiText-2 PPL | Eval Time (s) | Status |')
lines.append('|---|---|---|---|---|---|')
lines.append(f'| - | **BASELINE** | **{base_acc:.2f}%** | **{base_ppl:.2f}** | - | Reference |')

for i, r in enumerate(sorted_results, start=1):
    acc_s = f"{r['accuracy_pct']:.2f}%" if r.get('accuracy_pct') is not None else 'N/A'
    ppl_s = f"{r['perplexity']:.2f}" if r.get('perplexity') is not None else 'N/A'
    err_s = f"ERR: {r['error'][:25]}" if r.get('error') else 'OK'
    lines.append(f"| {i} | `{r['module']}` | {acc_s} | {ppl_s} | {r['duration_s']:.2f}s | {err_s} |")

with open('results/accuracy_and_ppl_benchmark_all131.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Successfully merged {len(sorted_results)} modules into results/accuracy_and_ppl_benchmark_all131.*")
