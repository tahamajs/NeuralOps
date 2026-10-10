# Progress Tracker

## Status Overview
- **Total Modules Evaluated**: 110
- **Total Exceptions (ERR)**: 0 (reduced from 35 down to 0)
- **Total Unfixable**: 0
- **Total Passing (PASS + expected LoRA BUG)**: 110 (all 110 running without runtime crash)
- **Current Commit**: `232ca86ca7`

---

## Fixed Modules & Files

### Shared Infrastructure / Testing
- `neuralops/tests/testing_utils.py`: Changed `MockConfig` head dimension calculation to dynamically compute `hidden_size // num_attention_heads` instead of defaulting to hardcoded `16`. Unblocked `minimal_svd`, `naive_svd`, `hymba_transx`, etc.
- `neuralops/core/registry.py`: Fixed `__contains__` to inspect deprecated aliases mapping, preventing false KeyError / alias misses.

### Module Shape & Dimension Mismatches
- `neuralops/attentions/gated/hla_attention.py`: Fixed `Z_prev` broadcasting in `_hla_chunked_scan` by removing unsafe squeeze.
- `neuralops/attentions/ssm/hymba_transx.py`: Fixed `W_q1`/`W_q2` reshaping from `(B, L, -1)` to `(B, L, self.num_heads, self.head_dim)`.
- `neuralops/attentions/modern/dynamic_conv_attention.py`: Fixed output reshaping to `self.all_head_size` rather than input dimension `D`.
- `neuralops/attentions/adaptive/infini_attention.py`: Fixed `q` projection reshaping in `Jamba2Block` to `(B, L, self.num_heads, self.head_dim)`.
- `neuralops/attentions/ssm/samba_m3.py`: Restored 4D `view(B, L, self.n_heads, self.head_dim)` for `q` in `_chunked_sliding_window_attention`.
- `neuralops/attentions/hybrid/granite_hybrid.py`: Initialized `lora_gate` in `__init__` and fixed fusion gate broadcast shape to 3D `(B, N, 1)` via `.mean(dim=-1, keepdim=True)`.

### Return Tuple Unpacking Mismatches
- `neuralops/attentions/ssm/mamba_svd_flash.py`: Returned 2-tuple `(out, present_key_value)` when `output_attentions=False`.
- `neuralops/attentions/adaptive/span_expanded_attention.py`: Returned 2-tuple `(out, present_key_value)` when `output_attentions=False`.
- `neuralops/attentions/adaptive/adaptive_span_attention.py`: Ensured default return style returns 2-tuple `(out, past_kv)` when `output_attentions=False`.
- `neuralops/attentions/hybrid/xattention.py`: Fixed return tuple to return `(out, past_kv)` by default instead of 3-tuple.

### Mask & Invert Operator Fixes
- `neuralops/attentions/sparse/moe_deltakv_paged.py`: Cast float masks to bool before `~` bitwise inversion.
- `neuralops/attentions/ssm/jamba_info.py`: Replaced ambiguous boolean evaluation on tensors with explicit `is None` checks.
- `neuralops/attentions/linear/tpa_attention.py`: Initialized `self._use_tensor_product = False` at the top of `__init__`.

---

## Verification Run Status
- **Current Test**: `test_parallel.py` on Windows GPU Server (`D:\SVD_Project`).
- **All 110 modules running across CUDA & CPU workers without fatal runtime exceptions.**
