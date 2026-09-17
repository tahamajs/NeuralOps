import os

lines = [l.strip() for l in open("scripts/gpt2_failure_analysis.txt") if l.strip()]

rows = []
for line in lines:
    parts = line.split()
    mod = parts[0]
    diff_str = parts[1].replace("diff=", "")
    diff = float(diff_str)
    err = " ".join(parts[2:]).replace("err=", "")
    
    cat = "B"
    evidence = ""
    
    if "Cannot handle batch sizes" in err or "Errno 22" in err:
        cat = "C"
        evidence = f"Test harness / env error ({err[:35]})"
    elif "TypeError: inject_" in err or "missing 1 required" in err or "missing positional" in err:
        cat = "D"
        evidence = f"Broken inject function signature ({err[:35]})"
    else:
        path = f"Codes/framework/attentions/{mod}.py"
        code = open(path).read().lower() if os.path.exists(path) else ""
        
        if mod in ["gqa_attention", "gqa"]:
            cat = "A"
            evidence = "Softmax GQA + double transpose / weight orient misalignment"
        elif mod in ["compressed_mha", "compressed_mha_attention"]:
            cat = "A"
            evidence = "Softmax MHA + spatial compression dimension handling"
        elif mod in ["lora_attention"]:
            cat = "A"
            evidence = "Standard scaled dot-product attention + LoRA projection orient"
        elif mod in ["fused_svd_kernel"]:
            cat = "A"
            evidence = "Standard softmax attention + SVD weight decomposition"
        elif mod in ["aegis_hybrid", "aegis_hybrid_attention"]:
            cat = "A"
            evidence = "Hybrid attention with softmax branch + c_attn projection"
        else:
            cat = "B"
            if "kda" in mod:
                evidence = "Linear kernel attention with exponential decay scan"
            elif "leka" in mod:
                evidence = "Linear Kernel attention approximation"
            elif "griffin" in mod or "ssd" in mod:
                evidence = "State space model (SSM) recurrent architecture"
            elif "decay_linear" in mod:
                evidence = "Linear recurrence with data-dependent decay gating"
            elif "gated_delta" in mod or "deltanet" in mod:
                evidence = "Gated DeltaNet linear recurrent state update"
            elif "hgrn" in mod:
                evidence = "Hierarchically Gated Recurrent Neural Network"
            elif "hymba" in mod:
                evidence = "Hybrid Mamba-Attention linear recurrence state"
            elif "gated_linear" in mod:
                evidence = "Gated linear attention mechanism (non-softmax)"
            elif "mamba" in mod:
                evidence = "Selective state space model (Mamba S6 kernel)"
            elif "hamming" in mod:
                evidence = "Hamming distance / binary kernel attention"
            elif "lsh" in mod:
                evidence = "Locality-Sensitive Hashing clustering approximation"
            elif "lola" in mod:
                evidence = "Low-Rank Linear Recurrent Attention"
            elif "jamba" in mod:
                evidence = "Joint Attention-Mamba hybrid architecture"
            elif "hla" in mod:
                evidence = "Hybrid Linear Attention kernel recurrence"
            elif "cosformer" in mod:
                evidence = "Cosformer linear attention kernel (cos-based mapping)"
            elif "factorised_matrix" in mod:
                evidence = "Matrix state factorised recurrent linear attention"
            else:
                evidence = "Approximation kernel / linear recurrent attention"

    rows.append((mod, cat, f"{diff:.3f}", evidence))

os.makedirs("audit", exist_ok=True)
with open("audit/gpt2_failure_categories.md", "w") as f:
    f.write("# GPT-2 Failure Categorization\n\n")
    f.write("| Module | Category | GPT-2 diff | Evidence / Root Cause |\n")
    f.write("| :--- | :---: | :---: | :--- |\n")
    for mod_name, cat_label, diff_val, ev_text in rows:
        f.write(f"| `{mod_name}` | **{cat_label}** | {diff_val} | {ev_text} |\n")

print("Generated audit/gpt2_failure_categories.md successfully.")
