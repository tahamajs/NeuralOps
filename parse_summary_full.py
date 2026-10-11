import glob, os, re, json

with open('_modules_tmp.json') as f:
    all_mods = json.load(f)
comp_names = [m['name'] for m in all_mods if m['type'] == 'compressor']

print('Total registered compressors in _modules_tmp.json:', len(comp_names))

triage = []
# buckets:
# CA — LoRA lora_B=0 (delta ≈ 0)
# CB — SVD rank == full (delta ≈ 0)
# CC — No real compression (bits >= 8, sparsity >= 1.0, keep_ratio >= 1.0)
# CD — Silent fallback (err > threshold → original weights)
# CE — Runtime exception (shape/dtype/index)
# CF — NaN/Inf in output
# CG — Timeout/OOM
# CH — KV-cache-only (no stationary weight change)
# PASS — delta > 1.0

for name in comp_names:
    log_file = f'results/delta_sweep/compressor_{name}.log'
    err_file = f'results/delta_sweep/compressor_{name}.err'
    json_file = f'results/delta_sweep/compressor_{name}.json'
    
    delta = None
    student_ppl = None
    teacher_ppl = 36.22
    
    log_content = ""
    if os.path.exists(log_file):
        with open(log_file, 'r', errors='ignore') as fp:
            log_content = fp.read()
            
    # Check delta from log
    # 1) [SANITY] Delta pre-baseline   = +345.89
    m_delta = re.findall(r'\[SANITY\]\s+Delta pre-baseline\s*=\s*([+-]?[0-9\.]+)', log_content)
    if m_delta:
        delta = float(m_delta[-1])
        
    m_ppl = re.findall(r'\[SANITY\]\s+Student pre-KD PPL\s*=\s*([0-9\.]+)', log_content)
    if m_ppl:
        student_ppl = float(m_ppl[-1])
    else:
        m_pre = re.findall(r'Pre:\s*([0-9\.]+)', log_content)
        if m_pre:
            student_ppl = float(m_pre[-1])
            if delta is None:
                delta = student_ppl - teacher_ppl
                
    err_content = ""
    if os.path.exists(err_file):
        with open(err_file, 'r', errors='ignore') as fp:
            err_content = fp.read()
            
    is_timeout = False
    if "exceeded 300s" in err_content or (os.path.exists(json_file) and "exceeded 300s" in open(json_file, errors='ignore').read()):
        is_timeout = True
        
    is_exception = False
    exc_match = re.search(r'Traceback \(most recent call last\):[\s\S]*?([A-Za-z0-9_]+Error:[^\n]+)', err_content + "\n" + log_content)
    if exc_match:
        is_exception = True
        exc_msg = exc_match.group(1)
    else:
        exc_msg = ""
        
    # Determine status & bucket
    if delta is not None and delta > 1.0 and not is_exception:
        status = "PASS"
        bucket = "PASS"
        cause = "OK (Delta > 1.0)"
        fix_strat = "None"
    elif is_timeout:
        status = "TIMEOUT"
        bucket = "CG"
        cause = "Exceeded 300s timeout"
        fix_strat = "Optimize calibration/LoRA or deprecate"
    elif is_exception:
        status = "EXCEPTION"
        bucket = "CE"
        cause = exc_msg
        fix_strat = "Fix shape/index/dtype error"
    elif name in ("aatc_xkv_compressor", "hybridkv_compressor", "liftr"):
        status = "FAIL"
        bucket = "CH"
        cause = "KV-cache only / dynamic activation"
        fix_strat = "Deprecate (weight KD cannot evaluate)"
    elif name == "faq_quantizer":
        status = "FAIL"
        bucket = "CC"
        cause = "default bits=8 (delta=+0.72 < 1.0)"
        fix_strat = "Force default bits=min(bits, 4)"
    elif name == "kronecker_tt_lora":
        status = "FAIL"
        bucket = "CA"
        cause = "delta = -0.01 (adapter perturbation too weak)"
        fix_strat = "Increase scaling/rank"
    elif name == "lillama_compressor":
        status = "FAIL"
        bucket = "CD"
        cause = "Deprecated alias / 0 layers compressed"
        fix_strat = "Deprecate"
    else:
        status = "FAIL"
        bucket = "CD"
        cause = f"Fallback to original weights (delta={delta:.2f})" if delta is not None else "Fallback / No layers compressed"
        fix_strat = "Relax error threshold / max_error"
        
    triage.append({
        'name': name,
        'status': status,
        'delta': delta,
        'student_ppl': student_ppl,
        'bucket': bucket,
        'cause': cause,
        'fix': fix_strat
    })

print(f"Total: {len(triage)}")
bucket_counts = {}
for t in triage:
    b = t['bucket']
    bucket_counts[b] = bucket_counts.get(b, 0) + 1
print("Bucket counts:", bucket_counts)

print("\n--- Non-PASS Modules ---")
for t in triage:
    if t['bucket'] != 'PASS':
        d_str = f"{t['delta']:+.2f}" if t['delta'] is not None else "N/A"
        print(f"| {t['name']} | {t['status']} | {d_str} | {t['bucket']} | {t['cause']} | {t['fix']} |")
