import glob, os, re, json

log_files = glob.glob('results/delta_sweep/compressor_*.log')
print(f'Total compressor logs: {len(log_files)}')

results = []
for f in log_files:
    mod_name = os.path.basename(f).replace('compressor_', '').replace('.log', '')
    with open(f, 'r', errors='ignore') as fp:
        content = fp.read()
    
    finished = ('SWEEP COMPLETELY FINISHED!' in content or 
                'ALL MODULES COMPLETE' in content or 
                'FINAL EVALUATION' in content)
    
    pre_match = re.search(r'Pre-KD Evaluation:[^\n]*PPL\s*=\s*([0-9\.]+)', content)
    post_match = re.search(r'Post-KD Evaluation:[^\n]*PPL\s*=\s*([0-9\.]+)', content)
    delta_match = re.search(r'Delta pre-baseline\s*=\s*([+-]?[0-9\.]+)', content)
    if not delta_match:
        delta_match = re.search(r'DELTA:\s*([+-]?[0-9\.]+)', content)
        
    error_match = re.search(r'Traceback \(most recent call last\):[\s\S]*?([A-Za-z0-9_]+Error:[^\n]+)', content)
    
    results.append({
        'name': mod_name,
        'finished': finished,
        'pre': pre_match.group(1) if pre_match else None,
        'post': post_match.group(1) if post_match else None,
        'delta': delta_match.group(1) if delta_match else None,
        'error': error_match.group(1) if error_match else None
    })

passed = [r for r in results if r['delta'] and float(r['delta']) >= 1.0]
zero_or_neg = [r for r in results if r['delta'] and float(r['delta']) < 1.0]
errored = [r for r in results if r['error'] and not r['finished']]
running = [r for r in results if not r['finished'] and not r['error']]

print(f'Evaluated logs: {len(results)}')
print(f'Passed (delta >= 1.0): {len(passed)}')
print(f'Zero/Neg delta: {len(zero_or_neg)}')
print(f'Errored: {len(errored)}')
print(f'Currently running: {len(running)}')

if zero_or_neg:
    print('\nZero/Negative delta:')
    for r in zero_or_neg:
        print(f"  {r['name']}: pre={r['pre']}, post={r['post']}, delta={r['delta']}")

if errored:
    print('\nErrored modules:')
    for r in errored:
        print(f"  {r['name']}: {r['error']}")

with open('_modules_tmp.json', 'r') as f:
    all_mods = json.load(f)
comp_names = [m['name'] for m in all_mods if m['type'] == 'compressor']
print(f'Total registered compressors in _modules_tmp.json: {len(comp_names)}')

evaluated_names = set(r['name'] for r in results)
untested = [name for name in comp_names if name not in evaluated_names]
print(f'Not yet started: {len(untested)}')
if untested:
    print('Remaining to start:', untested[:10])
