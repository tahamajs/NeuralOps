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
    # Highly vectorized PyTorch reference using two-stage tensor projection:
    # Step 1: u @ svd_vt.T -> [B, N, R]
    # Step 2: latent @ svd_u.T -> [B, N, D]
    # Step 3: fused elementwise state decay modulation -> [B, N, D]
    latent = torch.matmul(u, svd_vt.transpose(-1, -2))
    proj = torch.matmul(latent, svd_u.transpose(-1, -2))
    return decay * proj

def mamba_svd_flash(u: torch.Tensor, svd_u: torch.Tensor, svd_vt: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(u_in, svd_u_in, svd_vt_in, decay_in):
        mod = _get_module()
        return mod.mamba_svd_flash_fwd(u_in, svd_u_in, svd_vt_in, decay_in)
    return dispatch(cuda_fn, mamba_svd_flash_ref, mamba_svd_flash_ref, u, svd_u, svd_vt, decay, kernel_name="mamba_svd_flash")


