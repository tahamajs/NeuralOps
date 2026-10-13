"""Verify that README numbers match the actual registries and modules.

Run: python scripts/utils/verify_readme_counts.py
"""
import glob
import re
import sys
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

with open("README.md") as f:
    readme = f.read()

actual_attn = len(ATTENTION_REGISTRY)
actual_comp = len(COMPRESSOR_REGISTRY)
actual_kernels = len(glob.glob("neuralops/core/triton_kernels/*.py"))

attn_match = re.search(r"(\d+)\s+(?:canonical\s+)?attention mechanisms", readme)
comp_match = re.search(r"(\d+)\s+compression engines", readme)
kernel_match = re.search(r"(\d+)\s+Triton kernel modules", readme)

errors = []
if attn_match and int(attn_match.group(1)) != actual_attn:
    errors.append(f"Attention mismatch: README has {attn_match.group(1)}, actual is {actual_attn}")

if comp_match and int(comp_match.group(1)) != actual_comp:
    errors.append(f"Compressor mismatch: README has {comp_match.group(1)}, actual is {actual_comp}")

if kernel_match and int(kernel_match.group(1)) != actual_kernels:
    errors.append(f"Kernel mismatch: README has {kernel_match.group(1)}, actual is {actual_kernels}")

if errors:
    for e in errors:
        print(f"MISMATCH: {e}")
    sys.exit(1)

print(f"README counts verified successfully: {actual_attn} attention mechanisms, {actual_comp} compression engines, {actual_kernels} Triton kernels.")
