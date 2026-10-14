"""
Unified Custom Op and Torch.compile Compatibility Registry for NeuralOps Triton Kernels.

Provides seamless support for `torch.compile(model, fullgraph=False)` without graph breaks:
  1. Uses PyTorch >= 2.4 `torch.library.custom_op` and `register_fake` when available.
  2. Provides automatic FakeTensor / Meta shape propagation across all 26 kernel modules.
  3. Falls back gracefully on older PyTorch versions or non-CUDA environments.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import (
    linear_attn_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    state_update_pytorch_fallback,
    gqa_pytorch_fallback,
    triton_dispatch,
)
from neuralops.core.triton_kernels._types import (
    BLOCK_SIZE,
    MAX_TRITON_SEQ_LEN,
    SUPPORTED_HEAD_DIMS,
    check_triton_suitability,
)
from neuralops.core.triton_kernels.autotuner import Autotuner, autotune_kernel

logger = logging.getLogger("NeuralOps.OpRegistry")

# Detect PyTorch 2.4+ modern torch.library custom_op support
_HAS_TORCH_LIBRARY_CUSTOM_OP = hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_fake")


# ---------------------------------------------------------------------------
# Custom Op Registrations for torch.compile
# ---------------------------------------------------------------------------

if _HAS_TORCH_LIBRARY_CUSTOM_OP:
    try:
        # 1. Linear Attention Op
        @torch.library.custom_op("neuralops::linear_attn", mutates_args=())
        def _op_linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("linear_attn", q, k, v)

        @_op_linear_attn.register_fake
        def _op_linear_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(v)

        # 2. Sparse Attention Op
        @torch.library.custom_op("neuralops::sparse_attn", mutates_args=())
        def _op_sparse_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("sparse_attn", q, k, v)

        @_op_sparse_attn.register_fake
        def _op_sparse_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(q)

        # 3. Low Rank Op
        @torch.library.custom_op("neuralops::low_rank", mutates_args=())
        def _op_low_rank(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("low_rank", x, U, V)

        @_op_low_rank.register_fake
        def _op_low_rank_fake(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
            out_shape = list(x.shape[:-1]) + [V.shape[-1]]
            return x.new_empty(out_shape)

        # Register Autograd formulas for training / backward pass
        def _low_rank_bwd(ctx, grad_output):
            x, U, V = ctx.saved_tensors
            # y = (x @ V.T) @ U.T
            # dy/dx = grad_output @ U @ V
            grad_x = torch.matmul(torch.matmul(grad_output, U), V) if ctx.needs_input_grad[0] else None
            # dy/dU = grad_output.T @ (x @ V.T)
            x_2d = x.reshape(-1, x.shape[-1])
            g_2d = grad_output.reshape(-1, grad_output.shape[-1])
            h = torch.matmul(x_2d, V.t())
            grad_U = torch.matmul(g_2d.t(), h) if ctx.needs_input_grad[1] else None
            grad_V = torch.matmul(torch.matmul(g_2d, U).t(), x_2d) if ctx.needs_input_grad[2] else None
            return grad_x, grad_U, grad_V

        def _low_rank_setup_context(ctx, inputs, output):
            x, U, V = inputs
            ctx.save_for_backward(x, U, V)

        if hasattr(torch.library, "register_autograd"):
            try:
                torch.library.register_autograd(
                    "neuralops::low_rank", _low_rank_bwd, setup_context=_low_rank_setup_context
                )
            except Exception:
                pass


        # 4. GQA Op
        @torch.library.custom_op("neuralops::gqa", mutates_args=())
        def _op_gqa(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return triton_dispatch("gqa", k, v, num_kv_heads=num_kv_heads, num_heads=num_heads)

        @_op_gqa.register_fake
        def _op_gqa_fake(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
            ratio = num_heads // num_kv_heads
            k_out = k.repeat_interleave(ratio, dim=1) if k.dim() >= 3 and k.shape[1] == num_kv_heads else k
            v_out = v.repeat_interleave(ratio, dim=1) if v.dim() >= 3 and v.shape[1] == num_kv_heads else v
            return k_out, v_out

        # 5. GDN Scan Op
        @torch.library.custom_op("neuralops::gdn_scan", mutates_args=())
        def _op_gdn_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton
            if gdn_scan_triton is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
                try:
                    return gdn_scan_triton(q, k, v, beta, alpha)
                except Exception:
                    pass
            from neuralops.core.triton_kernels.gdn_scan import gdn_scan_torch
            return gdn_scan_torch(q, k, v, beta, alpha)

        @_op_gdn_scan.register_fake
        def _op_gdn_scan_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(v)

        # 6. GLA Scan Op
        @torch.library.custom_op("neuralops::gla_scan", mutates_args=())
        def _op_gla_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            from neuralops.core.triton_kernels.gla_scan import gla_scan_triton
            if gla_scan_triton is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
                try:
                    return gla_scan_triton(q, k, v, g)
                except Exception:
                    pass
            from neuralops.core.triton_kernels.gla_scan import gla_scan_torch
            return gla_scan_torch(q, k, v, g)

        @_op_gla_scan.register_fake
        def _op_gla_scan_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(v)

        # 7. HGRN2 Scan Op
        @torch.library.custom_op("neuralops::hgrn2_scan", mutates_args=())
        def _op_hgrn2_scan(f: torch.Tensor, iv: torch.Tensor, res_clip: float = 1.0) -> torch.Tensor:
            from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
            if hgrn2_scan_triton is not None and f.is_cuda and f.dtype in (torch.float16, torch.bfloat16):
                try:
                    return hgrn2_scan_triton(f, iv, res_clip=res_clip)
                except Exception:
                    pass
            from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_torch
            return hgrn2_scan_torch(f, iv, res_clip=res_clip)

        @_op_hgrn2_scan.register_fake
        def _op_hgrn2_scan_fake(f: torch.Tensor, iv: torch.Tensor, res_clip: float = 1.0) -> torch.Tensor:
            return torch.empty_like(iv)

        # 8. State Update Op
        @torch.library.custom_op("neuralops::state_update", mutates_args=())
        def _op_state_update(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("state_update", u, delta, A, B, C)

        @_op_state_update.register_fake
        def _op_state_update_fake(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(u)

        # 9. Softplus Op
        @torch.library.custom_op("neuralops::softplus", mutates_args=())
        def _op_softplus(x: torch.Tensor) -> torch.Tensor:
            from neuralops.core.triton_kernels.softplus import softplus_fwd
            if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
                try:
                    return softplus_fwd(x)
                except Exception:
                    pass
            return F.softplus(x)

        @_op_softplus.register_fake
        def _op_softplus_fake(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        # 10. Fused Norm Linear Op
        @torch.library.custom_op("neuralops::fused_norm_linear", mutates_args=())
        def _op_fused_norm_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float = 1e-5) -> torch.Tensor:
            normed = F.layer_norm(x, (x.shape[-1],), eps=eps)
            return F.linear(normed, weight, bias)

        @_op_fused_norm_linear.register_fake
        def _op_fused_norm_linear_fake(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float = 1e-5) -> torch.Tensor:
            out_shape = list(x.shape[:-1]) + [weight.shape[0]]
            return x.new_empty(out_shape)

        # 11. Recurrent Linear Step Op
        @torch.library.custom_op("neuralops::recurrent_linear_step", mutates_args=())
        def _op_recurrent_linear_step(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, kv_state: torch.Tensor, z_state: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
            return fused_recurrent_linear_step(q_t, k_t, v_t, kv_state, z_state, eps=eps)

        @_op_recurrent_linear_step.register_fake
        def _op_recurrent_linear_step_fake(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, kv_state: torch.Tensor, z_state: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = q_t.new_empty(q_t.shape[0], q_t.shape[1], v_t.shape[-1])
            return out, torch.empty_like(kv_state), torch.empty_like(z_state)

        # 12. Recurrent GLA Step Op
        @torch.library.custom_op("neuralops::recurrent_gla_step", mutates_args=())
        def _op_recurrent_gla_step(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, gk_t: torch.Tensor, kv_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            from neuralops.core.triton_kernels.state_update import fused_recurrent_gla_step
            return fused_recurrent_gla_step(q_t, k_t, v_t, gk_t, kv_state)

        @_op_recurrent_gla_step.register_fake
        def _op_recurrent_gla_step_fake(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, gk_t: torch.Tensor, kv_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out = q_t.new_empty(q_t.shape[0], q_t.shape[1], v_t.shape[-1])
            return out, torch.empty_like(kv_state)

        # 13. Recurrent GDN Step Op
        @torch.library.custom_op("neuralops::recurrent_gdn_step", mutates_args=())
        def _op_recurrent_gdn_step(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, g_t: torch.Tensor, beta_t: torch.Tensor, kv_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            from neuralops.core.triton_kernels.state_update import fused_recurrent_gdn_step
            return fused_recurrent_gdn_step(q_t, k_t, v_t, g_t, beta_t, kv_state)

        @_op_recurrent_gdn_step.register_fake
        def _op_recurrent_gdn_step_fake(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, g_t: torch.Tensor, beta_t: torch.Tensor, kv_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out = q_t.new_empty(q_t.shape[0], q_t.shape[1], v_t.shape[-1])
            return out, torch.empty_like(kv_state)

        # 14. FMS Scan Op
        @torch.library.custom_op("neuralops::fms_scan", mutates_args=())
        def _op_fms_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
            from neuralops.core.triton_kernels.fms_scan import fms_scan
            return fms_scan(q, k, v, decay)

        @_op_fms_scan.register_fake
        def _op_fms_scan_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(v)

        # 15. Selective Scan Op (Mamba)
        @torch.library.custom_op("neuralops::selective_scan", mutates_args=())
        def _op_selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: Optional[torch.Tensor] = None, delta_softplus: bool = True) -> torch.Tensor:
            from neuralops.core.triton_kernels.selective_scan import selective_scan_fn
            return selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=delta_softplus)

        @_op_selective_scan.register_fake
        def _op_selective_scan_fake(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: Optional[torch.Tensor] = None, delta_softplus: bool = True) -> torch.Tensor:
            return torch.empty_like(u)

        # 16. Cosformer Attention Op
        @torch.library.custom_op("neuralops::cosformer_attn", mutates_args=())
        def _op_cosformer_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, eps: float = 1e-4) -> torch.Tensor:
            from neuralops.core.triton_kernels.cosformer_triton_kernel import cosformer_triton_fwd
            return cosformer_triton_fwd(q, k, v, is_causal=is_causal, eps=eps)

        @_op_cosformer_attn.register_fake
        def _op_cosformer_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, eps: float = 1e-4) -> torch.Tensor:
            return torch.empty_like(v)

        # 17. Sliding Window Attention Op
        @torch.library.custom_op("neuralops::sliding_window_attn", mutates_args=())
        def _op_sliding_window_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 256) -> torch.Tensor:
            from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel
            return sliding_window_attention_kernel(q, k, v, window_size=window_size)

        @_op_sliding_window_attn.register_fake
        def _op_sliding_window_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 256) -> torch.Tensor:
            return torch.empty_like(q)

        # 18. Block Sparse Attention Op
        @torch.library.custom_op("neuralops::block_sparse_attn", mutates_args=())
        def _op_block_sparse_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            from neuralops.core.triton_kernels.block_sparse import block_sparse_attention
            return block_sparse_attention(q, k, v, mask)

        @_op_block_sparse_attn.register_fake
        def _op_block_sparse_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(q)

        # 19. Varlen Attention Op
        @torch.library.custom_op("neuralops::varlen_attn", mutates_args=())
        def _op_varlen_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, causal: bool = False) -> torch.Tensor:
            from neuralops.core.triton_kernels.varlen_attention import _varlen_attention_fwd
            return _varlen_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)

        @_op_varlen_attn.register_fake
        def _op_varlen_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, causal: bool = False) -> torch.Tensor:
            return torch.empty_like(q)

        # 20. Fused RoPE Op
        @torch.library.custom_op("neuralops::fused_rope", mutates_args=())
        def _op_fused_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            from neuralops.core.triton_kernels.fused_rope import fused_rope_embedding_kernel
            return fused_rope_embedding_kernel(q, k, cos, sin)

        @_op_fused_rope.register_fake
        def _op_fused_rope_fake(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(q), torch.empty_like(k)

        # 21. Fused SwiGLU Op
        @torch.library.custom_op("neuralops::fused_swiglu", mutates_args=())
        def _op_fused_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            from neuralops.core.triton_kernels.elementwise import fused_swiglu
            return fused_swiglu(x, gate)

        @_op_fused_swiglu.register_fake
        def _op_fused_swiglu_fake(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        # 22. Fused RMSNorm Op
        @torch.library.custom_op("neuralops::fused_rmsnorm", mutates_args=())
        def _op_fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            from neuralops.core.triton_kernels.elementwise import fused_rmsnorm
            return fused_rmsnorm(x, weight, eps=eps)

        @_op_fused_rmsnorm.register_fake
        def _op_fused_rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            return torch.empty_like(x)

        # 23. Fused Linear Cross Entropy Op
        @torch.library.custom_op("neuralops::fused_linear_ce", mutates_args=())
        def _op_fused_linear_ce(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor, ignore_index: int = -100, reduction: str = "mean", chunk_size: int = 1024) -> torch.Tensor:
            from neuralops.core.triton_kernels.fused_ce import fused_linear_cross_entropy
            return fused_linear_cross_entropy(hidden_states, weight, targets, bias=bias, ignore_index=ignore_index, reduction=reduction, chunk_size=chunk_size)

        @_op_fused_linear_ce.register_fake
        def _op_fused_linear_ce_fake(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor, ignore_index: int = -100, reduction: str = "mean", chunk_size: int = 1024) -> torch.Tensor:
            if reduction == "none":
                return hidden_states.new_empty(targets.shape)
            return hidden_states.new_empty(())

        # 24. Fused Linear KL Divergence (KD) Op
        @torch.library.custom_op("neuralops::fused_linear_kd", mutates_args=())
        def _op_fused_linear_kd(student_hidden: torch.Tensor, student_weight: torch.Tensor, student_bias: Optional[torch.Tensor], teacher_hidden: torch.Tensor, teacher_weight: torch.Tensor, teacher_bias: Optional[torch.Tensor], temperature: float = 2.0, reduction: str = "batchmean", chunk_size: int = 512) -> torch.Tensor:
            from neuralops.core.triton_kernels.fused_kd import fused_linear_kl_divergence
            return fused_linear_kl_divergence(student_hidden, student_weight, teacher_hidden, teacher_weight, student_bias=student_bias, teacher_bias=teacher_bias, temperature=temperature, reduction=reduction, chunk_size=chunk_size)

        @_op_fused_linear_kd.register_fake
        def _op_fused_linear_kd_fake(student_hidden: torch.Tensor, student_weight: torch.Tensor, student_bias: Optional[torch.Tensor], teacher_hidden: torch.Tensor, teacher_weight: torch.Tensor, teacher_bias: Optional[torch.Tensor], temperature: float = 2.0, reduction: str = "batchmean", chunk_size: int = 512) -> torch.Tensor:
            if reduction == "none":
                return student_hidden.new_empty(student_hidden.shape[:-1])
            return student_hidden.new_empty(())

        # 25. Paged Attention Op
        @torch.library.custom_op("neuralops::paged_attn", mutates_args=())
        def _op_paged_attn(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor, block_size: int = 16) -> torch.Tensor:
            from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel
            return paged_attention_kernel(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

        @_op_paged_attn.register_fake
        def _op_paged_attn_fake(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor, block_size: int = 16) -> torch.Tensor:
            return torch.empty_like(q)

        # 26. Paged Recurrent State Step Op
        @torch.library.custom_op("neuralops::paged_recurrent_state_step", mutates_args=())
        def _op_paged_recurrent_state_step(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, page_pool: torch.Tensor, block_tables: torch.Tensor, page_slot: int = 0, decay: float = 1.0) -> torch.Tensor:
            from neuralops.core.triton_kernels.paged_attention import paged_recurrent_state_step_triton
            return paged_recurrent_state_step_triton(q_t, k_t, v_t, page_pool, block_tables, page_slot=page_slot, decay=decay)

        @_op_paged_recurrent_state_step.register_fake
        def _op_paged_recurrent_state_step_fake(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, page_pool: torch.Tensor, block_tables: torch.Tensor, page_slot: int = 0, decay: float = 1.0) -> torch.Tensor:
            return torch.empty_like(q_t)

        _CUSTOM_OPS_REGISTERED = True
    except Exception as exc:
        logger.debug("Failed registering custom ops with torch.library: %s", exc)
        _CUSTOM_OPS_REGISTERED = False
else:
    _CUSTOM_OPS_REGISTERED = False


# ---------------------------------------------------------------------------
# High-Level Compilable Wrappers
# ---------------------------------------------------------------------------

def compilable_linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Torch.compile-friendly entrypoint for linear attention."""
    if _CUSTOM_OPS_REGISTERED and not kwargs:
        try:
            return torch.ops.neuralops.linear_attn(q, k, v)
        except Exception:
            pass
    return triton_dispatch("linear_attn", q, k, v, **kwargs)


def compilable_sparse_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Torch.compile-friendly entrypoint for sparse attention."""
    if _CUSTOM_OPS_REGISTERED and not kwargs:
        try:
            return torch.ops.neuralops.sparse_attn(q, k, v)
        except Exception:
            pass
    return triton_dispatch("sparse_attn", q, k, v, **kwargs)


def compilable_low_rank(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    """Torch.compile-friendly entrypoint for low rank factorization."""
    if _CUSTOM_OPS_REGISTERED and bias is None and not kwargs:
        try:
            return torch.ops.neuralops.low_rank(x, U, V)
        except Exception:
            pass
    return triton_dispatch("low_rank", x, U, V, bias=bias, **kwargs)


def register_custom_op_meta_shape(op_name: str, fake_fn: Callable) -> bool:
    """Register custom fake/meta implementation for torch.compile shape propagation."""
    if hasattr(torch.library, "register_fake"):
        try:
            torch.library.register_fake(op_name, fake_fn)
            return True
        except Exception as e:
            logger.debug("register_fake failed for %s: %s", op_name, e)
    return False


__all__ = [
    "compilable_linear_attn",
    "compilable_sparse_attn",
    "compilable_low_rank",
    "register_custom_op_meta_shape",
]
