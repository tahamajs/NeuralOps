# Kernels API Reference

NeuralOps implements **26 Triton JIT kernel modules** and complementary native CUDA extensions for subquadratic sequence modeling, paged state serving, and fused training objectives.

---

## 1. Triton Kernel Catalog (26 Canonical Modules)

| Module File | Capability | PyTorch Fallback | Autograd Support | Key Operators |
| :--- | :--- | :---: | :---: | :--- |
| `gdn_scan.py` | Gated DeltaNet scan | ✅ | ✅ | `gdn_scan_triton`, `gdn_scan_pytorch` |
| `gla_scan.py` | Chunked Gated Linear Attention | ✅ | ✅ | `chunk_gla_fwd`, `chunk_gla_bwd` |
| `hgrn2_scan.py` | High-order gated recurrent scan | ✅ | ✅ | `hgrn2_scan_triton`, `hgrn2_scan_pytorch` |
| `selective_scan.py` | Selective state-space scan (Mamba) | ✅ | ✅ | `selective_scan_triton` |
| `fms_scan.py` | Factorised matrix state update | ✅ | ✅ | `fms_recurrent_step_triton` |
| `paged_attention.py` | Non-contiguous virtual memory KV | ✅ | ❌ (Inference) | `paged_attention_kernel` |
| `flash_decoding.py` | Split-KV parallel decode attention | ✅ | ❌ (Inference) | `flash_decoding_stage1`, `flash_decoding_stage2` |
| `sliding_window.py` | Band-limited local context attention | ✅ | ✅ | `sliding_window_triton` |
| `fused_ce.py` | Fused linear projection + cross entropy| ✅ | ✅ | `fused_linear_cross_entropy` |
| `fused_kd.py` | Fused distillation KL-divergence | ✅ | ✅ | `fused_distillation_loss` |
| `fused_norm_linear.py`| Fused RMSNorm / LayerNorm + GEMM | ✅ | ✅ | `fused_norm_linear` |
| `fused_rope.py` | In-place rotary position encoding | ✅ | ✅ | `fused_rope_embedding_kernel` |
| `low_rank.py` | Fused SVD-QKV & W4A16 GEMM | ✅ | ✅ | `fused_svd_gemm_triton` |
| `elementwise.py` | Fast Walsh-Hadamard Transform (FWHT) | ✅ | ✅ | `fast_hadamard_transform` |
| `linear_attn.py` | Causal associative linear scan | ✅ | ✅ | `fused_chunk_linear_scan` |
| `cosformer_triton_kernel.py` | Cosformer feature projection | ✅ | ✅ | `cosformer_triton` |
| `block_sparse.py` | Dynamic block sparse attention tiles | ✅ | ✅ | `block_sparse_attention` |
| `sparse_attn.py` | Gather/scatter sparse attention | ✅ | ✅ | `topk_sparse_attention` |
| `gqa.py` | Grouped-Query attention expansion | ✅ | ✅ | `fused_gqa_triton` |
| `varlen_attention.py` | Ragged sequence batching | ✅ | ✅ | `varlen_attention_triton` |
| `state_update.py` | In-place recurrent hidden state step | ✅ | ✅ | `recurrent_state_update` |
| `softplus.py` | Numerically stable fused softplus | ✅ | ✅ | `fused_softplus` |
| `autotuner.py` | Hardware-aware tile autotuning | ✅ | N/A | `neuralops_autotune` |
| `hardware_profile.py` | SM capability and SRAM profiler | ✅ | N/A | `HardwareProfile` |
| `op_registry.py` | `torch.library` registration | ✅ | N/A | `register_kernel_op` |
| `_common.py` / `_types.py` | Abstract FakeTensor shape logic | ✅ | N/A | `should_use_triton` |

---

## 2. Using Triton Kernels

```python
import torch
from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton

# Input tensors: [Batch, Heads, Length, Dim]
q = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
beta = torch.sigmoid(torch.randn(2, 8, 2048, device="cuda", dtype=torch.float16))
alpha = torch.randn(2, 8, 2048, device="cuda", dtype=torch.float16)

# Execute fused associative recurrent scan
out = gdn_scan_triton(q, k, v, beta, alpha)
print("Output shape:", out.shape)
```
