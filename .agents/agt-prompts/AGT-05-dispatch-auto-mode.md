## AGT-05 — Fix dispatch auto-mode and guard the missing PyTorch fallback — Phase P0 — Findings F-27, F-28
**Depends on:** AGT-00 only
**Goal:** After this prompt, `auto` backend mode genuinely tries CUDA first, then Triton, then PyTorch; all 18 `dispatch_*` helpers have a real PyTorch level; and calling `dispatch()` (or any helper) without a PyTorch fallback raises a clear `RuntimeError` naming the kernel instead of crashing with `TypeError: 'NoneType' object is not callable` or a raw Triton `ImportError`.
**Context (Persian, 2-4 lines):** ادعای «دیسپچ سه‌سطحی» امروز روی مک دروغ است: حالت پیش‌فرض هرگز CUDA را امتحان نمی‌کند و helperها اصلاً سطح PyTorch ندارند، پس به‌جای fallback خطا می‌دهند. این دو باگ کوچک‌اند، اما دقیقاً روی ماشینی که کاربر دارد — CPU بدون CUDA و Triton — همه‌چیز را می‌شکنند.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are fixing the kernel dispatch helpers in exactly one file:
`neuralops/ops/cuda/_fallback.py` (760 lines), plus one new test file. Do not
touch the CUDA kernels, the Triton kernels, the catalogues, or any caller — only
the dispatch logic in this file and its tests.

## Background
Exact problems, with file paths and evidence (audit, 2026-11-06):
- `neuralops/ops/cuda/_fallback.py` ends with 18 helpers that are only
  CUDA → Triton (no PyTorch argument, no PyTorch level):
  `dispatch_direct_svd`, `dispatch_asvd`, `dispatch_fwsvd`, `dispatch_hnc_svd`,
  `dispatch_awq`, `dispatch_fp8_compressor`, `dispatch_kmeans_quantize`,
  `dispatch_wanda`, `dispatch_sparsegpt`, `dispatch_admm`,
  `dispatch_routing_attention`, `dispatch_lsh_attention`, `dispatch_cross_attention`,
  `dispatch_adaptive_sparse`, `dispatch_dynamic_conv`, `dispatch_token_compressed`,
  `dispatch_chela_slay`, `dispatch_rala`.
  These are exactly the compression / quantisation / pruning modules most likely
  to be called on a CPU-only machine — and they have no CPU path.
- The auto-mode bug, confirmed by grep: the code does
  `if backend == "auto": backend = _FORCED_BACKEND.get()`, and the default value
  of that `ContextVar` is `"auto"`, so the later check `backend == "cuda"` is
  `False`. CUDA is NEVER tried in `auto` mode; execution goes straight to Triton
  and, on a Mac without Triton, raises `ImportError`.
- `dispatch()` in the same file has no guard for a missing PyTorch reference:
  the string `pytorch_fn is None` occurs ZERO times in the file. The code does
  `if pytorch_ref is None and pytorch_fn is not None: pytorch_ref = pytorch_fn;
  pytorch_fn = pytorch_ref` and then calls `pytorch_fn(*args, **kwargs)` with no
  guard, at the end and in the CPU shortcut. If the caller supplied no PyTorch
  reference and both accelerated paths fail, the user gets
  `TypeError: 'NoneType' object is not callable` instead of a clear error.
- The correct pattern to copy is the three-level order already intended by the
  file itself and by `neuralops/core/device.py` (`UniversalDispatcher`):
  native/CUDA → Triton → PyTorch, raising a clear `RuntimeError` when no PyTorch
  fallback was supplied. Do not redesign dispatch — restore the intended order.
- The baseline from AGT-00 (`BASELINE.md`) records the pre-fix state. Read it
  first; your new tests are the before/after evidence here.

## Tasks
1. Read `neuralops/ops/cuda/_fallback.py` in full. Map, with exact line numbers
   and quoting the real code: (a) the `_FORCED_BACKEND` `ContextVar` and its
   default; (b) the `auto` resolution code; (c) the `dispatch()` function,
   including the aliasing block, the CPU shortcut, and every place `pytorch_fn`
   is called; (d) all 18 `dispatch_*` helpers, including the exact names of the
   internal CUDA and Triton functions each one calls. Do NOT assume the internal
   function names from any sketch — read them from the file.
2. Fix auto-mode. Add ONE shared resolver (for example `_resolve_backend`) that
   implements: `auto` → try CUDA first, then Triton, then PyTorch. Concretely,
   if `backend == "auto"`, resolve it from the forced backend when the forced
   value is not `"auto"`; otherwise resolve `"auto"` to `"cuda"` so the CUDA
   attempt actually happens. Route BOTH `dispatch()` and all 18 helpers through
   this single resolver — do not copy-paste the resolution logic 19 times.
   The correct order in every helper is: try CUDA (catch `ImportError` /
   `RuntimeError` and continue) → try Triton (catch `ImportError` /
   `AttributeError` and continue) → PyTorch (or the clear `RuntimeError` below).
3. Add a `pytorch_fn` parameter (default `None`) plus a final PyTorch level to
   EACH of the 18 helpers, mirroring `dispatch()`. The signature shape is
   `dispatch_<name>(*args, pytorch_fn=None, backend="auto", **kwargs)` — adapt
   to each helper's real signature as found in Task 1; do not change the
   positional arguments or their order. If a helper has no sensible PyTorch
   reference available anywhere in the file, it must raise the clear
   `RuntimeError` from Task 4 when both accelerated paths fail — never silently
   return `None` and never raise a raw Triton `ImportError` to the caller.
4. Guard `dispatch()`: immediately AFTER the aliasing block
   (`if pytorch_ref is None and pytorch_fn is not None: ...`), add:
   (code to add, indented):
       if pytorch_fn is None:
           raise RuntimeError(
               f"No PyTorch fallback supplied for kernel '{kernel_name}'; "
               "CUDA and Triton both unavailable"
           )
   This guard must sit BEFORE every call site of `pytorch_fn`, including the CPU
   shortcut — read the function again after editing and verify no unguarded call
   remains. Apply the same guard (with the helper's kernel name) at the end of
   each of the 18 helpers.
5. Do NOT change: the CUDA or Triton implementations themselves, the kernel
   catalogues, the `fla`-backend `IntegrationNotImplementedError` raises in this
   file (they are legitimate), or any caller. Do not add a new dependency.
6. Create `tests/test_ops_dispatch.py` (new file) with these tests (names exact):
   - `test_dispatch_helpers_auto_mode_tries_cuda_first`: mock the CUDA level and
     assert that in `auto` mode the CUDA implementation is attempted first.
   - `test_dispatch_helpers_fall_back_to_pytorch_on_cpu`: call EACH of the 18
     helpers on CPU (no CUDA, no Triton) with a `pytorch_fn` supplied, and assert
     each returns the PyTorch reference's result.
   - `test_dispatch_helpers_without_pytorch_fn_raise_runtime_error`: call each
     helper with no `pytorch_fn` and no CUDA/Triton available, and assert it
     raises `RuntimeError` (with the kernel name in the message) — NOT
     `ImportError` and NOT `TypeError`.
   - `test_dispatch_without_pytorch_fn_raises_runtime_error`: call `dispatch()`
     with no PyTorch fallback on CPU and assert it raises `RuntimeError` naming
     the kernel, not `TypeError: 'NoneType' object is not callable`.
   Each test must FAIL before the fix and PASS after it — state this explicitly
   in your report for each test. Tests must use mocks / CPU-only execution; they
   must not require a GPU, CUDA, or Triton.

## Constraints / do-not-do
- No silent fallbacks, no new dependencies, no breaking change without a
  migration note. (Adding a keyword parameter with default `None` is
  backwards-compatible; record it in `CHANGELOG.md` under the current version,
  2.0.6, as a one-line note.)
- Do not edit generated files, do not commit secrets, do not touch any file
  other than `neuralops/ops/cuda/_fallback.py`, `tests/test_ops_dispatch.py`
  and `CHANGELOG.md`.
- Do not swallow the accelerated-path exceptions silently in a way that hides
  them: catching `ImportError` / `RuntimeError` / `AttributeError` to fall
  through to the next level is the intended behaviour; catching bare `Exception`
  is not — keep the except clauses narrow, as in the sketch above.
- Do not run GPU-only tests; this Mac has no NVIDIA GPU and no `nvcc`.

## Tests to add/run
- Create: `tests/test_ops_dispatch.py` (tests named in Task 6).
- Run: `python -m pytest tests/test_ops_dispatch.py -v`
- Run: `python -m ruff check neuralops/ops/cuda/_fallback.py tests/test_ops_dispatch.py`
- Run: `grep -n "pytorch_fn is None" neuralops/ops/cuda/_fallback.py` —
  expected: at least one occurrence (the new guard), and the count must equal
  the number of guard sites you added (report the number).
- Run: `python -c "from neuralops.ops.cuda._fallback import dispatch, dispatch_direct_svd, dispatch_wanda; print('imports ok')"` — expected: imports succeed.
- Run (CPU check, no GPU needed):
  `python -c "from neuralops.ops.cuda._fallback import dispatch_wanda; print(dispatch_wanda(pytorch_fn=lambda *a, **k: 'pytorch-result'))"` — expected:
  prints `pytorch-result`. If the real signature of `dispatch_wanda` takes
  different arguments, adapt this one-liner to its real signature and report
  the exact command you ran.

## Verification (agent MUST run and report)
- `python -m pytest tests/test_ops_dispatch.py -v` — expected: all tests pass;
  report the pass count.
- The `grep -n "pytorch_fn is None"` count from Tests above.
- The CPU one-liner result from Tests above.
- `git diff --stat` and `git status --short` — confirm only
  `neuralops/ops/cuda/_fallback.py`, `tests/test_ops_dispatch.py` and
  `CHANGELOG.md` changed.
- Re-read `dispatch()` after editing and confirm: the auto resolution goes
  through the shared resolver, the guard sits before every `pytorch_fn` call,
  and the CUDA → Triton → PyTorch order is visible in the code.
- Checklist to tick: [ ] `auto` tries CUDA first [ ] all 18 helpers accept
  `pytorch_fn` [ ] all 18 helpers fall back to PyTorch on CPU [ ] no
  `pytorch_fn` call is unguarded [ ] missing fallback raises `RuntimeError`
  naming the kernel [ ] all new tests pass.

## Definition of done
- `auto` mode tries CUDA before Triton in `dispatch()` and in all 18 helpers.
- Every one of the 18 helpers has a `pytorch_fn` parameter and a final PyTorch
  level (or the clear `RuntimeError` when none was supplied).
- `dispatch()` with no PyTorch fallback raises `RuntimeError` naming the kernel,
  never `TypeError: 'NoneType' object is not callable`.
- On this CPU-only Mac, calling any helper with a `pytorch_fn` succeeds and
  calling it without one raises the clear `RuntimeError`.
- All new tests pass; each would have failed before the fix.

## Report back
- Files changed (exact paths).
- The exact `auto`-resolution code before and after (short quotes).
- Test results with pass/fail counts, and for each test why it would have failed
  before the fix.
- The `grep -n "pytorch_fn is None"` count and the CPU one-liner result.
- What you could NOT verify (e.g. behaviour on a real CUDA GPU — say so
  explicitly; the mocked tests are not a substitute).
- Any new finding (e.g. a helper whose CUDA/Triton function does not exist).

## Stop conditions
- If `neuralops/ops/cuda/_fallback.py` does not contain the `auto` bug or the
  18 helpers as described — STOP, quote what you actually found, and report it.
- If a helper's internal CUDA/Triton function name differs from what you
  expected, or a helper does not exist — STOP for that helper and report the
  exact discrepancy; do not invent a function name or a new helper.
- If a test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If a decision is needed (e.g. a helper has no PyTorch reference anywhere and
  one would have to be written from scratch) — STOP and report the decision
  needed; do not write a new PyTorch reference implementation inside this prompt.
```

---
