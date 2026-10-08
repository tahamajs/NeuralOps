#!/usr/bin/env python3
"""
Comprehensive Audit of Attention Modules for Causal/Decoder Correctness.

Audits all attention modules in neuralops/attentions/ and classifies each into:
  - TRUE_CAUSAL: Explicit chunked/recurrent causal logic or custom kernel (O(N) or O(N*C), no full softmax)
  - CAUSAL_VIA_MASK: Causal mask via tril/triu on softmax scores (O(N^2) softmax)
  - BROKEN_ON_DECODER: Uses softmax without causal mask in decoder mode (future token leakage)
  - ENCODER_ONLY: Bidirectional only (raises error or unsupported on decoder)
  - DENSE_FALLBACK: Has fallback to dense softmax when is_decoder/is_causal is True
"""

import ast
import csv
import os
import re
from pathlib import Path

ATTENTIONS_DIR = Path("neuralops/attentions")
OUTPUT_CSV = Path("results/audit/module_causal_audit.csv")

def analyze_module_file(filepath: Path) -> dict:
    stem = filepath.stem
    code = filepath.read_text(encoding="utf-8", errors="replace")

    # Patterns
    has_softmax = bool(re.search(r"\b(?:F\.)?softmax\s*\(", code, re.IGNORECASE))
    
    # Causal mask checks
    causal_mask_patterns = [
        r"torch\.tril",
        r"torch\.triu",
        r"causal_mask",
        r"attention_mask.*causal",
        r"triu.*diagonal\s*=\s*1",
        r"tril.*diagonal\s*=\s*0",
        r"masked_fill.*mask",
        r"is_causal.*True",
        r"fill_diagonal",
        r"band_diagonal",
    ]
    has_causal_mask = any(re.search(pat, code, re.IGNORECASE) for pat in causal_mask_patterns)

    # Chunked causal logic
    chunked_patterns = [
        r"_chunked_causal",
        r"_causal_linear_attention",
        r"chunk_size",
        r"num_chunks",
        r"chunked_delta",
        r"for\s+c\s+in\s+range.*chunk",
        r"range\(0,\s*[A-Za-z0-9_]+,\s*[A-Za-z0-9_]*chunk",
        r"causal_chunk",
    ]
    has_chunked_logic = any(re.search(pat, code, re.IGNORECASE) for pat in chunked_patterns)

    # Recurrent state checks
    recurrent_patterns = [
        r"\bS_running\b",
        r"\bstate\b.*\.cumsum",
        r"\bassociative_scan\b",
        r"\bcausal_scan\b",
        r"\bparallel_scan\b",
        r"\bselective_scan\b",
        r"\btriton.*scan\b",
        r"chunked_delta_rule",
        r"delta_rule",
        r"cumprod",
        r"cumsum.*dim\s*=\s*[-0-9]+",
        r"\bh_running\b",
        r"\bkv_state\b",
        r"\bssm\b",
        r"\bmamba\b",
        r"\bdecay_linear\b",
    ]
    has_recurrent_state = any(re.search(pat, code, re.IGNORECASE) for pat in recurrent_patterns)

    # Is decoder / causal aware
    decoder_aware_patterns = [
        r"self\.is_decoder",
        r"self\.is_causal",
        r"is_decoder\b",
        r"is_causal\b",
        r"causal\b",
    ]
    is_decoder_aware = any(re.search(pat, code) for pat in decoder_aware_patterns)

    # Dense fallback check (like nystrom had: `if self.is_decoder or self.is_causal: dense softmax`)
    dense_fallback_patterns = [
        r"if\s+(?:self\.)?(?:is_decoder|is_causal)[^:\n]*:\s*(?:#[^\n]*\n\s*)*return\s+.*(?:dense|softmax|F\.softmax)",
        r"if\s+(?:self\.)?(?:is_decoder|is_causal)[^:\n]*:\s*(?:#[^\n]*\n\s*)*[a-zA-Z0-9_]*\s*=\s*(?:F\.)?softmax",
        r"use_dense\s*=\s*.*(?:is_decoder|is_causal)",
        r"if\s+self\.is_decoder:\s*(?:#[^\n]*\n\s*)*[^\n]*softmax",
    ]
    has_dense_fallback = any(re.search(pat, code, re.IGNORECASE) for pat in dense_fallback_patterns)

    # Encoder-only check
    encoder_only_patterns = [
        r"if\s+(?:self\.)?is_decoder:\s*(?:#[^\n]*\n\s*)*raise\s+NotImplementedError",
        r"raise\s+NotImplementedError\([^)]*decoder[^)]*\)",
        r"assert\s+not\s+(?:self\.)?is_decoder",
        r"supports_causal\s*=\s*False",
        r"bidirectional\s+only",
    ]
    is_encoder_only = any(re.search(pat, code, re.IGNORECASE) for pat in encoder_only_patterns)

    # Specific known custom architectures
    is_true_linear_or_ssm = (
        has_chunked_logic 
        or has_recurrent_state 
        or "triton_svd" in stem 
        or "triton_cosformer" in stem
        or "gated_delta" in stem
        or "hgrn" in stem
        or "mamba" in stem
        or "selective_ssm" in stem
        or "chela_slay" in stem
        or "nystrom" in stem
        or "leka" in stem
        or "nexus" in stem
        or "regla" in stem
        or "linear_attention" in stem
        or "delta_net" in stem
    )

    # Classification
    notes = []
    if is_encoder_only:
        bucket = "ENCODER_ONLY"
        notes.append("Explicitly raises NotImplementedError or marked encoder-only")
    elif has_dense_fallback and not (has_chunked_logic and not has_softmax):
        bucket = "DENSE_FALLBACK"
        notes.append("Branch falls back to dense softmax when is_decoder/is_causal")
    elif is_true_linear_or_ssm and (not has_softmax or has_chunked_logic or has_recurrent_state):
        bucket = "TRUE_CAUSAL"
        notes.append("True causal recurrent, chunked, or scan linear/SSM attention")
    elif has_causal_mask and has_softmax:
        bucket = "CAUSAL_VIA_MASK"
        notes.append("Softmax attention with causal triangular mask (O(N^2))")
    elif has_softmax and not has_causal_mask:
        # Check if forward accepts attention_mask from outer GPT-2
        if "attention_mask" in code and ("q.shape" in code or "scores" in code):
            # If it applies attention_mask to scores, it could be causal via mask
            if re.search(r"scores\s*(?:\+|\+=)\s*attention_mask", code):
                bucket = "CAUSAL_VIA_MASK"
                notes.append("Applies external causal attention_mask to softmax scores")
            else:
                bucket = "BROKEN_ON_DECODER"
                notes.append("Uses softmax without explicit causal mask (future token leakage)")
        else:
            bucket = "BROKEN_ON_DECODER"
            notes.append("No causal masking found; unmasked softmax leaks future tokens")
    else:
        # Fallback inspection
        if has_causal_mask:
            bucket = "CAUSAL_VIA_MASK"
            notes.append("Applies causal masking")
        elif not is_decoder_aware:
            bucket = "ENCODER_ONLY"
            notes.append("Not decoder-aware; no causal handling implemented")
        else:
            bucket = "BROKEN_ON_DECODER"
            notes.append("Undefined causal behavior on decoder")

    return {
        "module_name": stem,
        "source_file": str(filepath),
        "bucket": bucket,
        "has_softmax": has_softmax,
        "has_causal_mask": has_causal_mask,
        "has_chunked_logic": has_chunked_logic,
        "has_recurrent_state": has_recurrent_state,
        "is_decoder_aware": is_decoder_aware,
        "notes": "; ".join(notes),
    }

def main():
    py_files = sorted(ATTENTIONS_DIR.glob("*.py"))
    py_files = [p for p in py_files if not p.stem.startswith("_")]

    print(f"Auditing {len(py_files)} attention modules in {ATTENTIONS_DIR}...")
    results = [analyze_module_file(p) for p in py_files]

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

    print(f"Saved audit to {OUTPUT_CSV}")

    # Counts
    counts = {}
    for r in results:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY REPORT")
    print("=" * 60)
    for b in ["TRUE_CAUSAL", "CAUSAL_VIA_MASK", "BROKEN_ON_DECODER", "ENCODER_ONLY", "DENSE_FALLBACK"]:
        print(f"  {b:20s}: {counts.get(b, 0):3d}")
    print(f"  Total modules audited: {len(results)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
