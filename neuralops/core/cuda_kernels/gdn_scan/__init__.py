"""
Gated DeltaNet Chunked Scan Hardware Operators.

Author: Taha Majlesi
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
