from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "gdn_delta_rule_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "gdn_delta_rule.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_gdn_delta_rule",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gdn_delta_rule_fwd"],
        )
    return _MODULE

def gdn_delta_rule_pytorch_ref(q, k, v, beta):
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan_pytorch
    alpha = torch.ones_like(beta)
    res = gdn_scan_pytorch(q, k, v, beta, alpha)
    return res[0] if isinstance(res, tuple) else res
def gdn_delta_rule_triton_ref(q, k, v, beta):
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton
    alpha = torch.ones_like(beta)
    return gdn_scan_triton(q, k, v, beta, alpha)

def gdn_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "gdn_delta_rule_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        gdn_delta_rule_triton_ref,
        gdn_delta_rule_pytorch_ref,
        q, k, v, beta,
        kernel_name="gdn_delta_rule",
    )
