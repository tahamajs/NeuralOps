import glob, json, os, re

files = glob.glob('results/delta_sweep/compressor_*.log')
print(f'Total compressor logs found: {len(files)}')

results = []
for f in files:
    mod = os.path.basename(f).replace('compressor_', '').replace('.log', '')
    try:
        cnt = open(f, encoding='utf-8', errors='ignore').read()
    except Exception:
        continue
    
    m_delta = re.search(r'Delta pre-baseline\s*=\s*([+-]?[0-9.]+)', cnt)
    m_ppl = re.search(r'Student pre-KD PPL\s*=\s*([0-9.]+)', cnt)
    finished = 'SWEEP COMPLETELY FINISHED' in cnt
    
    delta = float(m_delta.group(1)) if m_delta else None
    ppl = float(m_ppl.group(1)) if m_ppl else None
    
    if delta is not None and delta > 1.0 and finished:
        status = 'PASS'
    elif delta is not None and delta <= 1.0:
        status = 'DELTA_ZERO'
    elif not finished:
        status = 'IN_PROGRESS'
    else:
        status = 'ERROR'
    
    results.append({
        'module': mod,
        'status': status,
        'delta': delta,
        'ppl': ppl,
        'finished': finished
    })

passing = [r for r in results if r['status'] == 'PASS']
delta_zero = [r for r in results if r['status'] == 'DELTA_ZERO']
in_prog = [r for r in results if r['status'] == 'IN_PROGRESS']
errors = [r for r in results if r['status'] == 'ERROR']

print(f"\n--- Compressor Sweep Live Status ---")
print(f"Total Logged: {len(results)} / 73")
print(f"PASS (Delta > 1.0): {len(passing)}")
print(f"DELTA_ZERO (Delta <= 1.0): {len(delta_zero)}")
print(f"IN_PROGRESS: {len(in_prog)}")
print(f"ERROR: {len(errors)}")

if delta_zero:
    print(f"\nDelta Zero Modules:")
    for r in delta_zero:
        print(f"  {r['module']}: delta={r['delta']}, ppl={r['ppl']}")

if errors:
    print(f"\nError Modules:")
    for r in errors:
        print(f"  {r['module']}")

print("\nRecent 10 modules:")
for r in sorted(results, key=lambda x: x['module'])[-10:]:
    print(f"  {r['module']}: status={r['status']}, delta={r['delta']}, ppl={r['ppl']}")
