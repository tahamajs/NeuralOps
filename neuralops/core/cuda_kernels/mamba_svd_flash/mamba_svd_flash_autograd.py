from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "mamba_svd_flash_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "mamba_svd_flash.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_mamba_svd_flash", cpp_src, cuda_src, ["mamba_svd_flash_fwd"])
    return _MODULE

def mamba_svd_flash_ref(u: torch.Tensor, svd_u: torch.Tensor, svd_vt: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    # low-rank factorized linear transformation + state decay
    W = torch.matmul(svd_u, svd_vt)
    proj = F.linear(u, W)
    return decay * proj

def mamba_svd_flash(u: torch.Tensor, svd_u: torch.Tensor, svd_vt: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    cuda_fn = None
    if svd_u.shape[1] <= 128:
        def _cuda_fn(u, svd_u, svd_vt, decay):
            mod = _get_module()
            return mod.mamba_svd_flash_fwd(u.contiguous(), svd_u.contiguous(), svd_vt.contiguous(), decay.contiguous())
        cuda_fn = _cuda_fn
    return dispatch(cuda_fn, mamba_svd_flash_ref, mamba_svd_flash_ref, u, svd_u=svd_u, svd_vt=svd_vt, decay=decay, kernel_name="mamba_svd_flash")
