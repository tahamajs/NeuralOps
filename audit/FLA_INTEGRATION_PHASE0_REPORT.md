# FLA Integration Phase 0: Verification & Design Hardening Report

**Project:** NeuralOps (`tahamajs/NeuralOps`)  
**Mission:** Optional, Fail-Closed Upstream FLA (`flash-linear-attention`) Integration  
**Author:** AI Pair Programmer (verified per AGENTS.md §0.1, §0.2, §3.1, §10.2)  
**Status:** Phase 0 Complete — Awaiting Owner Sign-Off for Phase 1  

---

## 1. Premise Verification Table (§2.1)

Every claim in the preliminary proposal was subjected to exact POSIX static verification across the repository codebase:

| Premise from Source Doc | Verification Command | Actual Output / Result | Status |
| :--- | :--- | :--- | :---: |
| **P1: `fla` extra in `pyproject.toml`** | `grep -nA2 "\[project.optional-dependencies\]" pyproject.toml && grep -n "^fla = " pyproject.toml` | Line 46: `[project.optional-dependencies]`<br>Line 147: `fla = ["fla; sys_platform == 'linux'"]` | **CONFIRMED** |
| **P2: Existing `fla.ops` in layers** | `for f in neuralops/layers/...; do grep -nE "import fla\|from fla" "$f"; done` | • `linear/gla_attention.py:32`: `from fla.ops.gla import chunk_gla`<br>• `gated/flagship_fla.py:232,388,598`: `chunk_gdn2`, `parallel_path_attn`, `chunk_gdn`<br>• `linear/cosformer.py:435`: `from fla.ops.linear_attn import chunk_linear_attn`<br>• `gated/hgrn2_gated_outer.py:456`: `from fla.ops.gla import chunk_gla`<br>*(Note: `flagship_fla.py` is in `layers/gated/`, not `layers/linear/`)* | **CONFIRMED** (path corrected) |
| **P3: `fla` in `evaluation/harness.py`** | `grep -n "fla" neuralops/evaluation/harness.py` | Line 12: `Registers 'neuralops' and 'fla' model wrappers into lm_eval.`<br>Line 60: `@register_model("neuralops", "fla")` | **CONFIRMED** |
| **P4: `IntegrationNotImplementedError` class exists** | `grep -n "class IntegrationNotImplementedError" neuralops/core/errors.py` | Line 43: `class IntegrationNotImplementedError(NotImplementedError, NeuralOpsError):` | **CONFIRMED** |
| **P5: `force_kernel_backend` exists** | `grep -rn "def force_kernel_backend" neuralops/` | `neuralops/ops/cuda/_fallback.py:68`: `def force_kernel_backend(backend: str) -> Generator[None, None, None]:`<br>Supported backends: `'auto'`, `'cuda'`, `'triton'`, `'pytorch'`. (Note: `use_kernel_backend` does not exist; `force_kernel_backend` is the sole context manager). | **CONFIRMED** (context manager name locked) |
| **P6: 3-Level dispatch & `_AutogradBridge`** | `grep -rn "force_kernel_backend" neuralops/ops/` | Found in `_fallback.py`. Confirmed `_AutogradBridge(torch.autograd.Function)` line 110 handles fallback dispatch. | **CONFIRMED** |
| **P7: `UniversalAttentionWrapper` dispatch loop** | `grep -nE "_cached_dispatch_mode\|call_attempts" neuralops/utils/replace_attention.py` | Line 181: `call_attempts = [...]`<br>Line 228: `for mode_idx, attempt in call_attempts:`<br>Line 231: `self._cached_dispatch_mode = mode_idx` | **CONFIRMED** (caching exists, but naive wrapping risks triggering fallbacks) |

---

## 2. FLA Package API Discovery Status (§2.2)

POSIX inspection of `.venv` and the local filesystem:

```bash
$ ls .venv/lib/python*/site-packages/fla 2>/dev/null && echo "FLA INSTALLED" || echo "FLA NOT INSTALLED"
FLA NOT INSTALLED

$ find . -type d -name "fla*" -not -path "./.git/*" 2>/dev/null | head -10
./academic/paper/build_paper/flap
./academic/paper/build_paper/flat_llm
./academic/thesis/attentions_latex/modules/flap
... (No vendored fla package found)
```

**Honest Verification Finding (AGENTS.md §0.1):**
> **The FLA package API cannot be verified in this macOS environment.** Phase 1 must be built strictly fail-closed without inventing APIs. Phase 2 must begin with a remote-host API-discovery step (`pip show fla-linear-attention` and `python -c "import fla; print(dir(fla.layers))"` on `taha@192.168.1.100`) before any concrete layer-specific adapter code is written.

---

## 3. Forensic Diagnosis of the Naive Adapter (§2.3)

The draft adapter proposed in `docs/integrations/FLA_INTEGRATION_FEASIBILITY_AND_DESIGN.md` contains three critical runtime bugs that would break downstream models:

### Bug A — Return Tuple Arity Mismatch
* **Diagnosis:** The draft adapter unconditionally executed:
  ```python
  return attn_output, next_state
  ```
* **Failure Mode:** Hugging Face decoder blocks unpack attention outputs according to specific conventions:
  - `LlamaDecoderLayer` expects 3-tuple: `(attn_output, attn_weights, past_key_value)`.
  - `GPT2Block` expects 3-tuple: `(attn_output, present, attentions)`.
  - `BertSelfAttention` expects 2-tuple: `(context_layer, attention_probs)`.
* **Impact:** Every LLaMA, Mistral, Qwen, and GPT-2 forward call crashes with `ValueError: not enough values to unpack (expected 3, got 2)`.

### Bug B — `attention_mask` Silently Dropped
* **Diagnosis:** The draft `forward()` accepted `attention_mask`, but never passed it to `self.inner()`.
* **Failure Mode:** In encoder models (BERT, RoBERTa) and causal models with padding tokens, padding positions are not masked during attention score calculation.
* **Impact:** Numerical corruption across batched variable-length sequences.

### Bug C — `past_key_values` Semantics & Shape Mismatch
* **Diagnosis:** Standard Hugging Face softmax attention maintains KV cache tensors $K, V \in \mathbb{R}^{B \times H \times L \times D}$. Linear attention in FLA maintains an associative hidden state matrix $S \in \mathbb{R}^{B \times H \times d_k \times d_v}$ or $S \in \mathbb{R}^{B \times H \times D \times D}$.
* **Failure Mode:** Passing a 4D `(K, V)` tuple directly as `past_key_values=past_key_value` into FLA causes an immediate dimension mismatch or attribute error inside FLA's recurrent step.
* **Impact:** Complete failure in autoregressive text generation (`use_cache=True`).

---

## 4. Hardened Adapter Specification (§2.4)

To prevent runtime regressions, `neuralops/integrations/fla/adapter.py` will adhere to the following specification:

1. **Return-Arity Negotiation:**
   The adapter determines return arity dynamically from the parent model config or architecture:
   - For causal models (`llama`, `gpt2`, `mistral`, `qwen2`): returns `(attn_output, None, next_recurrent_state)`.
   - For encoder models (`bert`, `roberta`): returns `(attn_output, None)`.
   - Preserves compatibility with `output_attentions=True` (returns `None` for attention weights with an explicit debug log that linear attention does not compute $N \times N$ matrices).

2. **Cache Translation Layer (`cache_translator.py`):**
   - Implements `_translate_hf_cache_to_fla_state(past_key_value)` and `_translate_fla_state_to_hf_cache(next_state)`.
   - Validates state tensor dimensionality.
   - If an unsupported or incompatible cache type (e.g., legacy custom formats) is supplied, it **raises `IntegrationNotImplementedError`** with actionable diagnostic information. Silent degradation or fake cache return is strictly prohibited (AGENTS.md §0.2, §6.1).

3. **Mask Propagation & Verification:**
   - At initialization, the adapter inspects `inspect.signature(fla_layer_cls.forward)`.
   - If `attention_mask` is accepted by the underlying FLA layer, it is forwarded directly.
   - If not accepted, the adapter applies post-hoc zero-masking:
     `attn_output = attn_output * attention_mask.unsqueeze(-1)` (for 2D masks) or equivalent broadcasting.

4. **Fail-Closed on Unsupported Architectures:**
   - An explicit tuple `SUPPORTED_FAMILIES = ("llama", "gpt2", "mistral", "qwen2", "bert", "roberta")` guards surgery.
   - Passing an unverified model family raises `IntegrationNotImplementedError` rather than attempting an unsafe heuristic surgery.

5. **Strict Lazy Import & Zero Cold-Start Penalty:**
   - **No module-level `import fla` anywhere.**
   - Import of `fla` occurs strictly inside `FLALayerAdapter.__init__`.
   - `neuralops/integrations/fla/__init__.py` exposes `fla_is_available() -> bool` using `importlib.util.find_spec("fla")`, keeping cold-start overhead below 1ms.

---

## 5. Registry Bridge & Invariant Contract (§2.5)

To protect the canonical repository invariants (AGENTS.md §8):

1. **Separate Registry Dictionary:**
   - Canonical `ATTENTION_REGISTRY` remains locked at **exactly 131 entries**.
   - `EXTENSION_ATTENTION_REGISTRY: dict[str, type]` is introduced in `neuralops/integrations/fla/registry_bridge.py`.
   - All FLA-backed mechanisms are registered exclusively into `EXTENSION_ATTENTION_REGISTRY` with the mandatory `fla:` prefix (e.g., `fla:gla`, `fla:retnet`, `fla:gated_deltanet`, `fla:rwkv6`).

2. **Discovery API & Invariant Protection:**
   - `neuralops.list_attentions(include_extensions: bool = False)`:
     - When `include_extensions=False` (default): returns the canonical 131 entries.
     - When `include_extensions=True`: appends entries from `EXTENSION_ATTENTION_REGISTRY` (only if `fla_is_available()`).
   - `neuralops.get_attention_class(name)`:
     - If `name` starts with `fla:`: checks `EXTENSION_ATTENTION_REGISTRY`. If FLA is missing, raises `IntegrationNotImplementedError("Install via: pip install 'neuralops[fla]'")`.
     - Otherwise: queries canonical `ATTENTION_REGISTRY`.

3. **3-Level Dispatch Extension (Level 2b):**
   - The execution hierarchy in `_fallback.py` is augmented with `'fla'` as a valid backend:
     `SUPPORTED_BACKENDS = ("auto", "cuda", "triton", "fla", "pytorch")`.
   - When `force_kernel_backend("fla")` is active, the dispatcher routes to `fla.ops`.
   - If FLA is requested but not installed, it **raises `IntegrationNotImplementedError`** immediately.
   - Per AGENTS.md §7, every fallback transition from Level 2a (Triton) $\rightarrow$ Level 2b (FLA) $\rightarrow$ Level 3 (PyTorch) emits a descriptive `logger.warning`.

---

## 6. Version Bump Recommendation (§2.6)

* **Current Version:** `1.7.0` (released 2026-10-31).
* **Target Version:** **`1.8.0`** (Minor bump per SemVer 2.0.0 & AGENTS.md §14.2).
* **Rationale:** Adding `fla:*` extensions, `force_kernel_backend("fla")`, `list_fla_attentions()`, and `load_fla_weights()` introduces new backward-compatible public API functionality. It is NOT a patch fix.

---

## 7. Blockers & Decision Points for Owner Sign-Off

1. **Phase 1 Execution Permission:**
   Phase 1 builds the offline scaffold on macOS (`neuralops/integrations/fla/` + offline fail-closed tests). All offline tests verify that calling `fla:*` without `fla` installed raises `IntegrationNotImplementedError`. Owner approval is requested to proceed to Phase 1.
2. **Version Pinning Target:**
   Current `pyproject.toml` has unpinned `fla; sys_platform == 'linux'`. In Phase 2, we recommend pinning to `fla-linear-attention>=0.2.0,<0.3.0` once confirmed on the remote host.

---

**Phase 0 Status:** COMPLETED. No production code written. Awaiting owner sign-off to begin Phase 1.
