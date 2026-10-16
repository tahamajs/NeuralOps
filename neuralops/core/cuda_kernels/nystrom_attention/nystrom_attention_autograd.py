from __future__ import annotations
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "nystrom_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "nystrom_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_nystrom_attention", cpp_src, cuda_src, ["nystrom_attn_fwd"])
    return _MODULE

def nystrom_attention_ref(q_landmarks: torch.Tensor, k_landmarks: torch.Tensor, v_landmarks: torch.Tensor, pinv_landmarks: torch.Tensor) -> torch.Tensor:
    # A_hat = S_tilde @ pinv(K_tilde) @ V_tilde
    return torch.matmul(torch.matmul(q_landmarks, pinv_landmarks), v_landmarks)

def nystrom_attention(q_landmarks: torch.Tensor, k_landmarks: torch.Tensor, v_landmarks: torch.Tensor, pinv_landmarks: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q_landmarks, k_landmarks, v_landmarks, pinv_landmarks):
        mod = _get_module()
        return mod.nystrom_attn_fwd(q_landmarks.contiguous(), k_landmarks.contiguous(), v_landmarks.contiguous(), pinv_landmarks.contiguous())
    return dispatch(cuda_fn, nystrom_attention_ref, nystrom_attention_ref, q_landmarks, k_landmarks=k_landmarks, v_landmarks=v_landmarks, pinv_landmarks=pinv_landmarks, kernel_name="nystrom_attention")
