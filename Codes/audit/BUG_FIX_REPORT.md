# Bug Fix Report

## Group A — Critical (12 fixes)
| Fix | File | Before | After | Status |
|-----|------|--------|-------|--------|
| A1 | `a3_compressor.py` | `max_error=0.20` | `max_error=0.50` | DONE (Prev) |
| A2 | `a3_compressor.py` | hardcoded `True` | `bool(kwargs.get(...))` | DONE (Prev) |
| A3 | `a3_compressor.py` | `torch.cuda.OutOfMemoryError` | fallback `_CUDA_OOM` | DONE (Prev) |
| A4 | `a3_compressor.py` | GPU first always | Configurable CPU fallback | DONE (Prev) |
| A5 | `come.py` | Ignored T5 | Added `model.encoder.block` etc. | DONE |
| A6 | `come.py` | Copied low-rank weights directly | `SVDLinear` insertion | DONE |
| A7 | `sdrm_compressor.py`| Protected all heads | Split out attn heads, optional config | DONE |
| A8 | `sdrm_compressor.py`| Silent return if rank>max | Logs warning & clamps `min_rank` | DONE |
| A9 | `utils/model_family.py`| `absolute` for T5 | `relative` for T5 | DONE |
| A10 | `run_quality_sweep.py`| `ThreadPoolExecutor` | `subprocess.Popen` isolation | DONE |
| A11 | `run_quality_sweep.py`| Benchmark before `merge()`| Added `m.merge()` fallback loop | DONE |
| A12 | `run_quality_sweep.py`| `model.config.tokenizer = ...` | Removed mutations | DONE |

## Group B — Major (11 fixes)
| Fix | File | Before | After | Status |
|-----|------|--------|-------|--------|
| B1 | `a3_compressor.py` | `0.5 * min(in, out)` | Break-even rank ceiling | DONE |
| B2 | `a3_compressor.py` | Hardcoded mlp list | `FFN_PATTERNS` dictionary | DONE |
| B3 | `a3_compressor.py` | `logger.warning(e)` | `logger.exception(e)` | DONE |
| B4 | `come.py` | Returning 768 on unknown | Raises clear `ValueError` | DONE |
| B5 | `come.py` | `ModuleList` / `list` only | Added `nn.Sequential` | DONE |
| B6 | `sdrm_compressor.py`| Silent SDRM fallback | Handled safely inside `compress_linear`| SKIP/N/A |
| B7 | `run_quality_sweep.py`| Unreachable code | Control flow verified intact | SKIP/N/A |
| B8 | `run_quality_sweep.py`| `strict=False` silent | `LOG.warning` on unexpected/missing keys | DONE |
| B9 | `run_quality_sweep.py`| `except: pass` | Replaced with explicit `LOG.debug` | DONE |
| B10 | `run_quality_sweep.py`| Static timeouts | Subprocess auto-kills correctly | DONE (via A10) |
| B11 | `run_quality_sweep.py`| OOM limited to some models | Applied globally to all models | DONE |

## Group C — Minor (7 fixes)
| Fix | File | Before | After | Status |
|-----|------|--------|-------|--------|
| C1 | `a3_compressor.py` | Duplicate `hasattr` checks | Merged checking logic via `mod_with_nf` | DONE |
| C2 | `a3_compressor.py` | `r_ceiling` missing from stats | Propagated via `stats.update()` | DONE |
| C3 | `come.py` | `last_hidden_state` only | Handles `hidden_states` tuples | DONE |
| C4 | `come.py` | Random dummy input tokens | Uses `bos_token_id` | DONE |
| C5 | `sdrm_compressor.py`| Crashed if config missing | Fallbacks to `getattr(module, ...)` | DONE |
| C6 | `run_quality_sweep.py`| Unused stats tests | Skipped (avoid mutating parent on-the-fly) | SKIP |
| C7 | `run_quality_sweep.py`| Single seed | Skipped (avoid mutating parent on-the-fly) | SKIP |

## Verification
- All compressors import: **YES**
- Git Remote Synchronized: **YES**
- Smoke test: Currently running on the GPU server.
