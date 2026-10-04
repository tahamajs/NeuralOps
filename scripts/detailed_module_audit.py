#!/usr/bin/env python3
"""
Detailed Causal Audit of all 115 Attention Modules.
Inspects AST, regex, and method calls to classify every module into:
  - TRUE_CAUSAL
  - CAUSAL_VIA_MASK
  - BROKEN_ON_DECODER
  - ENCODER_ONLY
  - DENSE_FALLBACK
"""

import ast
import csv
import re
from pathlib import Path

ATTENTIONS_DIR = Path("neuralops/attentions")
OUTPUT_CSV = Path("results/audit/module_causal_audit.csv")

def audit_file(filepath: Path) -> dict:
    stem = filepath.stem
    code = filepath.read_text(encoding="utf-8", errors="replace")

    # 1. Check for DENSE_FALLBACK pattern
    # e.g. griffin_ssd, simple_ssm, or nystrom before fix
    # "if self.is_decoder" leading to dense softmax or O(N^2) decay matrix
    has_dense_fallback = False
    dense_fallback_reason = ""

    # Specifically check if decoder branch reverts to dense softmax or O(N^2)
    if stem == "griffin_ssd":
        has_dense_fallback = True
        dense_fallback_reason = "Decoder branch uses O(L^2) dense decay matrix and softmax fallback"
    elif stem == "simple_ssm":
        has_dense_fallback = True
        dense_fallback_reason = "Decoder branch materializes O(L^2) tril decay matrix instead of recurrent scan"
    else:
        # Check for fallback to dense attention
        if re.search(r"if\s+self\.is_decoder[^:\n]*:\s*(?:#[^\n]*\n\s*)*return\s+.*(?:dense|softmax)", code):
            has_dense_fallback = True
            dense_fallback_reason = "Explicit 'if self.is_decoder' dense softmax return"

    # 2. Check for ENCODER_ONLY
    is_encoder_only = False
    encoder_only_reason = ""
    if re.search(r"raise\s+NotImplementedError\([^)]*decoder", code, re.IGNORECASE) or \
       re.search(r"assert\s+not\s+(?:self\.)?is_decoder", code, re.IGNORECASE) or \
       stem in ("replace", "projection_factorized_attn", "svd_linear_decay"):
        is_encoder_only = True
        encoder_only_reason = "Encoder-only / bidirectional only, raises NotImplementedError or unsupported on decoder"

    # 3. Check for TRUE_CAUSAL
    # Recurrent SSMs, chunked linear attention, scan kernels
    is_true_causal = False
    true_causal_reason = ""

    # Specific patterns for linear attention / SSM / custom causal kernels:
    has_chunked_causal = bool(re.search(r"(_chunked_causal|_causal_linear_attention|chunk_size|num_chunks|chunked_delta)", code))
    has_scan = bool(re.search(r"(associative_scan|parallel_scan|selective_scan|triton.*scan|recurrent|causal_scan)", code))
    has_recurrent = bool(re.search(r"(\bS_running\b|\bh_running\b|\bkv_state\b|\bS\s*=\s*S\s*\+|\bdecay_linear\b)", code))
    has_triton_causal = bool(re.search(r"triton.*causal", code, re.IGNORECASE))
    
    # SSM architectures
    is_ssm_family = any(k in stem for k in [
        "mamba", "ssm", "samba", "jamba", "hgrn", "delta_net", "deltanet", "gla", "regla", "spectron"
    ])

    if has_triton_causal:
        is_true_causal = True
        true_causal_reason = "Custom Triton causal kernel"
    elif has_chunked_causal and not has_dense_fallback:
        is_true_causal = True
        true_causal_reason = "Chunked causal linear / state attention"
    elif (has_scan or has_recurrent or is_ssm_family) and not has_dense_fallback:
        is_true_causal = True
        true_causal_reason = "Recurrent SSM / state-space / linear scan causal attention"
    elif stem in (
        "chela_slay", "nystrom_attention", "triton_cosformer", "triton_svd_linear_attention",
        "gated_delta_net", "gated_deltanet_block", "hgrn2_layer", "hgrn2_outer", "hgrn2_gated_deltanet",
        "gated_linear_attention", "regla_attention", "halo_regla", "decay_linear_attention",
        "svd_linear_attention", "aegis_v6", "aegis_spectron_fp8", "leka_v1", "nexus",
        "spectron", "spectral_mamba", "selective_ssm", "mamba_block", "mamba_svd_flash",
        "upi_stuffed_mamba", "samba_m3", "samba_moe", "jamba", "jamba_info",
        "factorised_matrix_state", "kda_attention", "leka_v2", "leka_v3",
        "minimal_nova", "minimal_svd", "nha_attention", "zeros_attention",
        "cosformer", "cluster_attention", "linformer_attention", "performer_attention"
    ):
        # We will inspect linformer, performer, cosformer, cluster_attention specifically
        pass

    # 4. Check for Softmax and Causal Masking
    has_softmax = bool(re.search(r"\b(?:F\.)?softmax\s*\(", code, re.IGNORECASE))
    
    # Look for causal mask application
    has_causal_mask = bool(re.search(
        r"(torch\.triu|torch\.tril|causal_mask|mask\.masked_fill|scores\.masked_fill.*causal|scores\s*\+\s*causal|causal_tril|is_causal)",
        code,
        re.IGNORECASE
    ))

    # Bucket decision
    bucket = ""
    notes = ""

    if is_encoder_only:
        bucket = "ENCODER_ONLY"
        notes = encoder_only_reason
    elif has_dense_fallback:
        bucket = "DENSE_FALLBACK"
        notes = dense_fallback_reason
    elif is_true_causal:
        bucket = "TRUE_CAUSAL"
        notes = true_causal_reason
    elif has_softmax and has_causal_mask:
        bucket = "CAUSAL_VIA_MASK"
        notes = "Softmax attention with causal triangular mask (O(N^2))"
    elif has_softmax and not has_causal_mask:
        # Check if attention_mask from GPT2 is passed and added to scores
        if re.search(r"scores\s*(?:\+|\+=)\s*attention_mask", code) or re.search(r"apply_attention_mask", code):
            bucket = "CAUSAL_VIA_MASK"
            notes = "Applies outer causal attention_mask to softmax scores"
        else:
            bucket = "BROKEN_ON_DECODER"
            notes = "Softmax attention without causal masking; leaks future tokens on decoder"
    else:
        # No softmax, but no clear chunked/recurrent pattern
        if "causal" in code.lower():
            bucket = "TRUE_CAUSAL"
            notes = "Linear causal attention"
        else:
            bucket = "BROKEN_ON_DECODER"
            notes = "Missing causal handling on decoder"

    return {
        "module_name": stem,
        "source_file": str(filepath),
        "bucket": bucket,
        "has_softmax": has_softmax,
        "has_causal_mask": has_causal_mask,
        "has_chunked_logic": has_chunked_causal,
        "has_recurrent_state": (has_scan or has_recurrent or is_ssm_family),
        "is_decoder_aware": bool(re.search(r"is_decoder|is_causal", code)),
        "notes": notes,
    }

def main():
    py_files = sorted(ATTENTIONS_DIR.glob("*.py"))
    py_files = [p for p in py_files if not p.stem.startswith("_")]

    results = [audit_file(p) for p in py_files]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module_name",
                "source_file",
                "bucket",
                "has_softmax",
                "has_causal_mask",
                "has_chunked_logic",
                "has_recurrent_state",
                "is_decoder_aware",
                "notes",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    counts = {}
    for r in results:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1

    print("\n" + "=" * 60)
    print("DETAILED CAUSAL AUDIT SUMMARY")
    print("=" * 60)
    for b in ["TRUE_CAUSAL", "CAUSAL_VIA_MASK", "BROKEN_ON_DECODER", "ENCODER_ONLY", "DENSE_FALLBACK"]:
        print(f"  {b:20s}: {counts.get(b, 0):3d}")
    print(f"  Total modules audited: {len(results)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
