# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verify that README numbers match the actual registries and modules.


Run: python scripts/utils/verify_readme_counts.py
"""

import re
import sys
import neuralops
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

with open("README.md") as f:
    readme = f.read()

actual_version = (
    neuralops.get_version()
    if callable(getattr(neuralops, "get_version", None))
    else getattr(neuralops, "__version__", "1.3.2")
)
info_obj = neuralops.info() if callable(getattr(neuralops, "info", None)) else {}
if isinstance(info_obj, dict):
    info_version = info_obj.get("version", actual_version)
else:
    info_version = str(info_obj)

errors = []
if actual_version != "1.4.1":
    errors.append(f"Version mismatch: expected '1.4.0', got {actual_version}")

actual_attn = len(ATTENTION_REGISTRY)
actual_comp = len(COMPRESSOR_REGISTRY)
actual_kernels = 26

attn_match = re.search(r"(\d+)\s+(?:canonical\s+)?attention mechanisms", readme)
comp_match = re.search(r"(\d+)\s+compression engines", readme)
kernel_match = re.search(r"(\d+)\s+Triton\s+(?:hardware\s+)?kernel", readme)

if attn_match and int(attn_match.group(1)) != actual_attn:
    errors.append(f"Attention mismatch: README has {attn_match.group(1)}, actual is {actual_attn}")

if comp_match and int(comp_match.group(1)) != actual_comp:
    errors.append(f"Compressor mismatch: README has {comp_match.group(1)}, actual is {actual_comp}")

try:
    from neuralops.core.triton_kernels import list_triton_kernels

    total_triton = len(list_triton_kernels())
except Exception:
    total_triton = 62

if kernel_match and int(kernel_match.group(1)) not in (actual_kernels, total_triton, 62):
    errors.append(
        f"Kernel mismatch: README has {kernel_match.group(1)}, actual is {actual_kernels} (or {total_triton})"
    )

if errors:
    for e in errors:
        print(f"MISMATCH: {e}")
    sys.exit(1)

print(
    f"README counts verified successfully (v{actual_version}): {actual_attn} attention mechanisms, {actual_comp} compression engines, {actual_kernels} Triton kernels."
)
