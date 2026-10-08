# Dead Functions & Classes Audit (Round 2)

## Confirmed Dead Items (0 external references)

| File:Line | Item Type | Name | Confidence |
| :--- | :--- | :--- | :---: |
| `neuralops/analysis/module_analysis/aggregators.py:130` | unused variable | `sub_cat` | 100% |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:385` | unused variable | `pretrained_qkv` | 100% |
| `neuralops/attentions/svd/salt_attention.py:973` | unused variable | `grad_in` | 100% |
| `neuralops/attentions/svd/triton_svd_linear_attention.py:318` | unused variable | `stride_dob` | 100% |
| `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py:305` | unused variable | `new_seq_length` | 100% |
| `neuralops/core/memory.py:86` | unused variable | `exc_tb` | 100% |
| `neuralops/core/memory.py:86` | unused variable | `exc_val` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:15` | unused variable | `MidLse` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:15` | unused variable | `MidOut` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:19` | unused variable | `stride_mb` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:19` | unused variable | `stride_mc` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:19` | unused variable | `stride_md` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:19` | unused variable | `stride_mh` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:20` | unused variable | `stride_lb` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:20` | unused variable | `stride_lc` | 100% |
| `neuralops/core/triton_kernels/flash_decoding.py:20` | unused variable | `stride_lh` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:14` | unused variable | `OUT_K` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:17` | unused variable | `stride_cos_b` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:18` | unused variable | `stride_sin_b` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:20` | unused variable | `stride_okb` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:20` | unused variable | `stride_okd` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:20` | unused variable | `stride_okh` | 100% |
| `neuralops/core/triton_kernels/fused_rope.py:20` | unused variable | `stride_okn` | 100% |
| `neuralops/core/triton_kernels/paged_attention.py:15` | unused variable | `KCACHE` | 100% |
| `neuralops/core/triton_kernels/paged_attention.py:15` | unused variable | `VCACHE` | 100% |
| `neuralops/core/triton_kernels/paged_attention.py:17` | unused variable | `stride_kblk` | 100% |
| `neuralops/core/triton_kernels/paged_attention.py:17` | unused variable | `stride_kidx` | 100% |
| `neuralops/core/triton_kernels/paged_attention.py:18` | unused variable | `stride_vblk` | 100% |
| `neuralops/core/triton_kernels/paged_attention.py:18` | unused variable | `stride_vidx` | 100% |
| `neuralops/data/synthetic_generator.py:79` | unused variable | `transition_temp` | 100% |

## Uncertain Items (Referenced Externally)

| File:Line | Item Type | Name | Confidence | External Refs |
| :--- | :--- | :--- | :---: | :---: |
| `neuralops/__init__.py:28` | unused variable | `spec` | 100% | 2 |
| `neuralops/attentions/__init__.py:114` | unused variable | `spec` | 100% | 2 |
| `neuralops/compression_engine/__init__.py:44` | unused variable | `spec` | 100% | 2 |
| `neuralops/core/triton_kernels/varlen_attention.py:13` | unused variable | `max_seqlen_q` | 100% | 1 |
| `neuralops/core/triton_kernels/varlen_attention.py:14` | unused variable | `max_seqlen_k` | 100% | 1 |

## Recommendations

- **Kernel Parameters / Strides**: In Triton kernels (`paged_attention`, `flash_decoding`, `fused_rope`), keep parameter signatures matching CUDA/Triton standard conventions or prefix unused arguments with `_`.
- **Context Managers**: `exc_val`, `exc_tb` in `__exit__` methods are required by Python context manager protocol.
- **Class / Function Invariants**: Zero confirmed dead top-level public functions or classes were found; core library modularity is intact.
