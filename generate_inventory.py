import glob
import os
import re

targets = [
    # FAIL_DELTA
    ("compressor", "aatc_xkv_compressor", "FAIL_DELTA"),
    ("compressor", "faq_quantizer", "FAIL_DELTA"),
    ("compressor", "fwsvd_compressor", "FAIL_DELTA"),
    ("compressor", "hashed_net_compressor", "FAIL_DELTA"),
    ("compressor", "huffman_coder", "FAIL_DELTA"),
    ("compressor", "hybridkv_compressor", "FAIL_DELTA"),
    ("compressor", "kronecker_tt_lora", "FAIL_DELTA"),
    ("compressor", "liftr", "FAIL_DELTA"),
    ("compressor", "lillama_compressor", "FAIL_DELTA"),
    ("compressor", "lsc_compressor", "FAIL_DELTA"),
    ("compressor", "mgie_budget", "FAIL_DELTA"),
    ("compressor", "residual_svd_compressor", "FAIL_DELTA"),
    ("compressor", "roast_hashed_linear", "FAIL_DELTA"),
    ("compressor", "rotational_slicegpt_int4", "FAIL_DELTA"),
    ("compressor", "rpca_compressor", "FAIL_DELTA"),
    ("compressor", "saes_svd_v2", "FAIL_DELTA"),
    ("compressor", "salr", "FAIL_DELTA"),
    ("compressor", "slicegpt", "FAIL_DELTA"),
    ("compressor", "soka_compressor", "FAIL_DELTA"),
    ("compressor", "subspace_projection", "FAIL_DELTA"),
    ("compressor", "svd_linear", "FAIL_DELTA"),
    ("compressor", "swift_svd_act", "FAIL_DELTA"),
    ("compressor", "tomoe_compressor", "FAIL_DELTA"),
    ("compressor", "tt_compressor", "FAIL_DELTA"),
    ("compressor", "uniql_compressor", "FAIL_DELTA"),
    # ERROR
    ("compressor", "mamba_shedder", "ERROR"),
    # UNKNOWN
    ("attention", "adaptive_span_attention", "UNKNOWN"),
    ("compressor", "slicegpt_pca", "UNKNOWN"),
    ("compressor", "svd_llm_compressor", "UNKNOWN"),
]

results = []

for mod_type, mod_name, group_status in targets:
    log_file = f"results/delta_sweep/{mod_type}_{mod_name}.log"
    err_file = f"results/delta_sweep/{mod_type}_{mod_name}.err"
    
    log_content = ""
    err_content = ""
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
    if os.path.exists(err_file):
        with open(err_file, "r", encoding="utf-8", errors="ignore") as f:
            err_content = f.read()
            
    # Extract delta
    delta = None
    m_delta = re.search(r"Delta pre-baseline\s*=\s*([+-]?\d+\.?\d*)", log_content)
    if m_delta:
        delta = m_delta.group(1)
        
    # Extract root cause
    root_cause = "Unknown"
    
    # Check for tracebacks first
    full_text = log_content + "\n" + err_content
    if "Traceback (most recent call last):" in full_text:
        # find the exception line
        tb_lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        for i in range(len(tb_lines)-1, -1, -1):
            if "Error:" in tb_lines[i] or "Exception:" in tb_lines[i]:
                root_cause = tb_lines[i]
                break
        if root_cause == "Unknown":
            root_cause = "Traceback during execution"
    elif "DEPRECATED" in log_content:
        root_cause = "Module marked DEPRECATED in registry/aliases"
    elif "exceeded" in full_text and ("threshold" in full_text or "Keeping original" in full_text or "fallback" in full_text):
        root_cause = "All layers exceeded error threshold; fell back to original dense weights"
    elif "0 activations linearized, 0 attentions wrapped" in log_content:
        root_cause = "No target activation or attention modules matched (no-op)"
    elif "avg_adapter_delta=0.0003" in log_content or "delta_w is the adapter delta" in log_content:
        root_cause = "Base weights frozen and unchanged; LoRA/TT delta initialized near zero"
    elif "without mutating stationary weights" in log_content or "Dynamic KV Cache" in log_content:
        root_cause = "Dynamic KV-cache only; stationary weights uncompressed"
    elif "0 compressed" in log_content:
        root_cause = "Zero layers compressed (filter/fallback skipped all)"
    elif "Student pre-KD PPL <= teacher baseline PPL (delta=+0.00" in log_content:
        root_cause = "Stationary weights identical to teacher baseline (delta=0.0)"
    elif not log_content and not err_content:
        root_cause = "No log output produced"
        
    results.append({
        "module": mod_name,
        "type": mod_type,
        "status": group_status,
        "delta": delta if delta is not None else "N/A",
        "root_cause": root_cause,
        "log_tail": "\n".join(log_content.strip().splitlines()[-10:]) if log_content else "EMPTY",
        "err_tail": "\n".join(err_content.strip().splitlines()[-10:]) if err_content else "EMPTY"
    })

print(f"Extracted info for {len(results)} target modules.")

for r in results:
    print(f"=== {r['type']}:{r['module']} ({r['status']}, delta={r['delta']}) ===")
    print(f"Root Cause: {r['root_cause']}")
    if r['status'] in ('ERROR', 'UNKNOWN'):
        print(f"Err tail:\n{r['err_tail']}")
    print()
