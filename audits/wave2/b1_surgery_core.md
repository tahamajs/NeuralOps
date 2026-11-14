# Agent B1 (Surgery Core) Wave 2 Audit & Remediation Report

**Date:** 2026-11-15  
**Subsystem:** Surgery Core Engine  
**Agent ID:** B1 (`subagent-Agent-B1--Surgery-Core`)  
**Status:** Complete & Fully Verified  

---

## 1. Domain Scope & Invariants

Agent B1 audited and hardened the eight core surgery modules:
- `neuralops/surgery/director.py`
- `neuralops/surgery/fx_surgery.py`
- `neuralops/surgery/scanner.py`
- `neuralops/surgery/endoskeletal_shell.py`
- `neuralops/surgery/layer_replacement.py`
- `neuralops/surgery/attention_replace.py`
- `neuralops/surgery/model_surgery.py`
- `neuralops/surgery/__init__.py`

*Protected Exclusions (unmodified per orchestration instructions):*
`surgery/mla_surgery.py`, `moa_router.py`, `streaming_distill.py`, `shadow_twin.py`.

---

## 2. Targeted Bug Remediations

### P0-6: `director.py::certify_model` — Elimination of Hardcoded `True` Invariant Laws
- **Issue:** 8 out of 10 invariant laws in `UniversalSurgeryDirector.certify_model` were hardcoded to `True` (or bypassed via `has_cross or True`, `is_mla_routed or self.profile.is_mla`), allowing broken models, missing bridges, or non-finite weights to falsely report 10/10 compliance.
- **Fix:** Replaced hardcoded stubs with rigorous verification logic across all Ten Inviolable Laws:
  - **Law 1 (QK-Norm Preservation):** Formally validates presence of `q_norm`, `k_norm`, or `has_qk_norm` modules on configured models.
  - **Law 2 (Logit Soft-Capping Temperature Folding):** Validates bounded folded scale factor $0 < \tau_{\text{folded}} < \infty$ and verifies that replaced linear bridges match the target model's `softcapping_factor`.
  - **Law 3 (Zero-Bleed Padding Mask Sanitization):** Tests dummy inputs with padded masks against `sanitize_kv_for_padding_mask` and `sanitize_output_for_padding_mask` to guarantee that padded positions evaluate strictly to $0.0$.
  - **Law 4 (Static Cross-Attention Caching):** Enforces static cross-attention bridges (`CrossAttentionLinearBridge` or `prefill_encoder`) on encoder-decoder models (e.g. T5/BART), removing the `or True` bypass.
  - **Law 5 (RoPE Pre-Projection Alignment):** Enforces even `head_dim` constraint for paired rotary dimensions and verifies rotation identity on zero-angle inputs.
  - **Law 6 (Decoupled Latent State Routing):** Enforces `DeepSeekMLASurgeryBridge` or latent state initialization for DeepSeek MLA architectures, removing the `or self.profile.is_mla` bypass.
  - **Law 7 (Causal State Monotonicity & Exponential State Decay):** Verifies decay factor $\gamma \in (0, 1)$ across all linear bridges and guarantees finite Frobenius norm bound $\frac{c}{1 - \gamma}$.
  - **Law 8 (Tied Embedding Preservation):** Checks shared pointer storage and memory identity between input and output embeddings.
  - **Law 9 (FP32 Accumulation Guard):** Enforces that `last_accum_dtype` and `recurrent_state_dtype` across surgical modules are strictly `torch.float32`.
  - **Law 10 (Master Universal Surgery Integration):** Asserts model is an `nn.Module`, contains non-empty parameters, has zero non-finite (NaN/Inf) weights, and satisfies all prior laws with zero violations.

### P0-10: `surgery/__init__.py::apply_backbone_surgery_hooks` — Real Runtime Forward Adapter
- **Issue:** `apply_backbone_surgery_hooks` was not exported in `surgery/__init__.py`, and the underlying implementation was a no-op merely toggling `supports_gqa` and `supports_rope` flags without registering forward adapters. When foundation model decoders passed keyword arguments (`position_ids`, `position_embeddings`, `past_key_value`, `use_cache`) or expected a 3-tuple return `(attn_out, attn_weights, past_kv)`, injected attention modules would throw `TypeError`.
- **Fix:** 
  - Implemented forward method wrapping via `_make_hooked_forward` in `neuralops/surgery/attention_replace.py`.
  - Adapts arguments according to inspect signature, provides fallback paths for minimal arguments `(hidden_states, attention_mask=...)`, and guarantees a 3-tuple return `(attn_output, attn_weights, past_key_value)`.
  - Exported `apply_backbone_surgery_hooks` in `neuralops/surgery/__init__.py` and `__all__`.
  - Removed top-level import of `neuralops.api` from `neuralops/surgery/__init__.py`, resolving circular import issues and adhering to AGENTS.md Rule 1.3 (no import-time side effects).

### P0-19: `fx_surgery.py::causal_prefix_scan` — Bounded $O(C^2 + D^2)$ Memory Scaling
- **Issue:** `causal_prefix_scan` allocated a 5-dimensional tensor `kv` of shape `[B, H, T, D, D]` via einsum `bhtd,bhte->bhtde` to perform `torch.cumsum`. For sequences of length $T=4096, D=128$, this resulted in multi-gigabyte memory allocations and out-of-memory errors.
- **Fix:**
  - Implemented chunkwise parallel associative scan with bounded block size $C=128$.
  - Intra-chunk attention is evaluated via $(L, L)$ local causal mask ($L \le C$).
  - Inter-chunk context is propagated via running recurrent state $S \in \mathbb{R}^{B \times H \times D \times E}$ and normalizer vector $z \in \mathbb{R}^{B \times H \times D \times 1}$ with zero $T$ memory dependency.
  - Supports exponential state decay $\gamma \in (0, 1)$ according to Law 7 with inter-chunk decay factor $\gamma^L$.
  - Preserves exact mathematical parity with unchunked cumulative sums down to $10^{-5}$ tolerance.

### P1: `endoskeletal_shell.py::apply_moa_schedule` — Dict and Empty Attribute Resilience
- **Issue:** Custom models utilizing `nn.ModuleDict` or dictionaries for layer blocks crashed during fallback inspection or in-place transplantation (`spec.attr_name.isdigit()` failed and `setattr` on `dict` raised `AttributeError`).
- **Fix:**
  - Added support for `(nn.ModuleList, list, tuple, nn.ModuleDict, dict)` containers during candidate spec discovery.
  - Implemented safe dictionary assignment `parent_module[attr] = bridge` for dict-like parents and dynamic resolution when `spec.attr_name` is empty or unpopulated.

### P1: `layer_replacement.py::_instantiate_replacement` — Error Propagation
- **Issue:** Nested `try ... except TypeError:` blocks silently swallowed `TypeError` exceptions originating from bugs within custom attention `__init__` methods, masking real errors and attempting to call unparameterized constructors `new_attention_cls()`.
- **Fix:**
  - Implemented `_safe_call_factory` utilizing `inspect.signature` to filter keyword arguments matching the constructor.
  - Distinguishes argument mismatch at call time from exceptions originating inside the factory body using traceback inspection, guaranteeing proper error propagation.

---

## 3. Test Verification & Code Quality

### Test Results
Executed regression and unit test suites:
```bash
./.venv/bin/pytest -o addopts= -p no:django -p no:cov -p no:sugar \
  tests/surgery/test_surgery_core_wave2_regression.py \
  tests/surgery/test_replace_attention.py \
  tests/surgery/test_surgery_exhaustive.py -v
```
- `tests/surgery/test_surgery_core_wave2_regression.py`: 13 passed in 3.74s
- `tests/surgery/test_replace_attention.py`: 2 passed
- `tests/surgery/test_surgery_exhaustive.py`: 22 passed
- **Total: 37 passed in 4.10s (100% green)**

### Linting
Executed Ruff check across all domain files:
```bash
./.venv/bin/ruff check \
  neuralops/surgery/director.py \
  neuralops/surgery/fx_surgery.py \
  neuralops/surgery/scanner.py \
  neuralops/surgery/endoskeletal_shell.py \
  neuralops/surgery/layer_replacement.py \
  neuralops/surgery/attention_replace.py \
  neuralops/surgery/model_surgery.py \
  neuralops/surgery/__init__.py \
  tests/surgery/test_surgery_core_wave2_regression.py
```
- **Result:** `All checks passed!` (0 errors, 0 warnings).
