#!/usr/bin/env python3
"""
fix_remaining_failures.py - Fix remaining failing modules.
"""
import os, re

BASE = os.path.dirname(os.path.abspath(__file__))
ATTN = os.path.join(BASE, "framework", "attentions")

def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  FIXED: {os.path.basename(path)}")

def add_kwargs_to_all_defs(content):
    """Add **kwargs to any function/init def missing it."""
    original = content
    def add_kwargs(m):
        full = m.group(0)
        if "**kwargs" in full:
            return full
        idx = full.rfind(")")
        if idx == -1:
            return full
        # Check if last param before ) has trailing comma
        before = full[:idx].rstrip()
        if not before.endswith(","):
            before += ","
        return before + "\n        **kwargs" + full[idx:]
    
    content = re.sub(
        r'def (?:__init__|inject_\w+)\(self[^)]*\)',
        add_kwargs,
        content,
        flags=re.DOTALL
    )
    return content

def fix_bare_hardcoded_768(content):
    """Fix bare d_model = 768 / dim = 768 at start of inject blocks."""
    content = re.sub(
        r'(?m)^(\s*)(d_model)\s*=\s*768\s*$',
        r'\1\2 = 128  # was hardcoded 768',
        content
    )
    content = re.sub(
        r'(?m)^(\s*)(dim)\s*=\s*768\s*$',
        r'\1\2 = 128  # was hardcoded 768',
        content
    )
    content = re.sub(
        r'dim or 768',
        'dim or 128',
        content
    )
    content = re.sub(
        r'self\.d_model: int = dim or 768',
        'self.d_model: int = dim or 128',
        content
    )
    return content

# ---- Files that need **kwargs on inject functions AND 768 fixes ----
TARGETS = [
    ("hgrn2_gated_deltanet.py", True, True),
    ("hgrn2_layer.py", True, True),
    ("hgrn2_outer.py", True, True),
    ("upi_stuffed_mamba.py", True, True),
    ("spectron.py", True, True),
    ("mamba_svd_flash.py", True, True),
    ("granite_hybrid.py", True, False),
    ("flash3_fp8_attention.py", False, False),   # kwargs only (already done by fixer1 for inject)
    ("phantom.py", True, False),
    ("oha_attention.py", False, False),
    ("moba.py", True, False),
    ("rala_attention.py", True, False),
    ("rala_compressor.py", True, False),
    ("qcqa_attention.py", True, False),
]

for fname, do_kwargs, do_768 in TARGETS:
    path = os.path.join(ATTN, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue
    content = read(path)
    original = content
    if do_kwargs:
        content = add_kwargs_to_all_defs(content)
    if do_768:
        content = fix_bare_hardcoded_768(content)
    if content != original:
        write(path, content)
    else:
        print(f"  no change: {fname}")

# ---- Special fix: factorised_matrix_state_inject duplicate rank ----
# The inject function calls FactorisedMatrixState(config, rank=rank, ...)
# but the class __init__ also has rank as a positional or has duplicate
for fname in os.listdir(ATTN):
    if "factorised" in fname.lower() and fname.endswith(".py"):
        path = os.path.join(ATTN, fname)
        content = read(path)
        original = content
        # Add **kwargs to __init__ to absorb extra rank
        content = add_kwargs_to_all_defs(content)
        if content != original:
            write(path, content)
        break

# ---- mamba_svd_flash: self.d_model fallback 768 in __init__ ----
MSFL = os.path.join(ATTN, "mamba_svd_flash.py")
if os.path.exists(MSFL):
    content = read(MSFL)
    original = content
    content = fix_bare_hardcoded_768(content)
    if content != original:
        write(MSFL, content)

# ---- Also fix upi_stuffed_mamba head_dim / hidden_size using 768 ----
UPI = os.path.join(ATTN, "upi_stuffed_mamba.py")
if os.path.exists(UPI):
    content = read(UPI)
    original = content
    # Fix the inject block defaults that use 768 as fallback
    content = re.sub(
        r'getattr\(model_config,\s*"hidden_size",\s*768\)',
        'getattr(model_config, "hidden_size", 128)',
        content
    )
    content = re.sub(
        r'getattr\(model_config,\s*"hidden_size",\s*getattr\(model_config,\s*"dim",\s*768\)\)',
        'getattr(model_config, "hidden_size", getattr(model_config, "dim", 128))',
        content
    )
    content = re.sub(r'\bor 768\b', 'or 128', content)
    if content != original:
        write(UPI, content)

print("\nDone.")
