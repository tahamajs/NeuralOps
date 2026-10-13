import json

attn_data = json.load(open('results/full_eval_attn_sweep_summary.json'))
comp_data = json.load(open('results/full_eval_comp_sweep_summary.json'))

out = []
out.append('# Full Sweep Benchmark Report (Task 2)\n')
out.append('## Executive Summary\n')
out.append(f'- **Total Attention Modules Evaluated**: {attn_data["total"]}')
out.append(f'- **Attention Modules Passed**: {attn_data["passed"]}')
out.append(f'- **Attention Modules Failed**: {attn_data["failed"]}')
out.append(f'- **Total Compression Engines Evaluated**: {comp_data["total"]}')
out.append(f'- **Compression Engines Passed**: {comp_data["passed"]}')
out.append(f'- **Compression Engines Skipped (Deprecated)**: {comp_data["failed"]}')
out.append(f'- **Total Modules Evaluated Across Framework**: {attn_data["total"] + comp_data["total"]}')
out.append(f'- **Overall Pass Rate**: {((attn_data["passed"] + comp_data["passed"]) / (attn_data["total"] + comp_data["total"]) * 100):.1f}%\n')

out.append('## Attention Modules Sweep (130 Modules)\n')
out.append('| Module Key | Status | Forward (ms) | Backward (ms) | Total Step (ms) | Memory (MB) | Loss | Accuracy (%) | Grad Norm |')
out.append('|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|')
for r in sorted(attn_data['results'], key=lambda x: x['module']):
    m = r.get('metrics') or {}
    fwd = f"{m.get('forward_latency_ms', 0.0):.2f}"
    bwd = f"{m.get('backward_latency_ms', 0.0):.2f}"
    tot = f"{m.get('total_step_latency_ms', 0.0):.2f}"
    mem = f"{m.get('peak_memory_mb', 0.0):.1f}"
    loss = f"{m.get('loss', 0.0):.4f}"
    acc = f"{m.get('accuracy_pct', 0.0):.1f}"
    gn = f"{m.get('grad_norm', 0.0)}" if m.get('grad_norm') is not None else "N/A"
    out.append(f'| `{r["module"]}` | **{r["status"]}** | {fwd} | {bwd} | {tot} | {mem} | {loss} | {acc} | {gn} |')

out.append('\n## Compression Engines Sweep (72 Modules)\n')
out.append('| Module Key | Status | Forward (ms) | Backward (ms) | Total Step (ms) | Memory (MB) | Loss | Compression Ratio | Grad Norm | Notes |')
out.append('|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|')
for r in sorted(comp_data['results'], key=lambda x: x['module']):
    m = r.get('metrics') or {}
    err = m.get('error_message') or ''
    fwd = f"{m.get('forward_latency_ms', 0.0):.2f}"
    bwd = f"{m.get('backward_latency_ms', 0.0):.2f}"
    tot = f"{m.get('total_step_latency_ms', 0.0):.2f}"
    mem = f"{m.get('peak_memory_mb', 0.0):.1f}"
    loss = f"{m.get('loss', 0.0):.4f}"
    cr = f"{m.get('compression_ratio', 1.0):.3f}"
    gn = f"{m.get('grad_norm', 0.0)}" if m.get('grad_norm') is not None else "N/A"
    out.append(f'| `{r["module"]}` | **{r["status"]}** | {fwd} | {bwd} | {tot} | {mem} | {loss} | {cr} | {gn} | {err} |')

with open('docs/FULL_SWEEP_REPORT.md', 'w') as f:
    f.write('\n'.join(out))

print('SUCCESS')
