# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated DeltaNet Chunked Scan Hardware Operators.
"""

from neuralops.core.cuda_kernels.gdn_scan.gdn_autograd import (
    chunked_delta_gate_scan,
    chunked_delta_gate_scan_pytorch_ref,
)

gdn_scan = chunked_delta_gate_scan
gdn_scan_pytorch_ref = chunked_delta_gate_scan_pytorch_ref

__all__ = [
    "chunked_delta_gate_scan",
    "chunked_delta_gate_scan_pytorch_ref",
    "gdn_scan",
    "gdn_scan_pytorch_ref",
]
