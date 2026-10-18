# Benchmark Suite Hygiene & Reliability Audit Report

**Author**: Taha Majlesi  
**Date**: October 19, 2026  
**Package Version**: 1.3.7  
**Hardware Target**: NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9, CUDA 12.8)  

---

## 1. Executive Summary

An exhaustive hygiene and reliability audit was performed across all benchmark entrypoints (`benchmarks/`) and execution scripts (`scripts/benchmarks/`, `scripts/analysis/`, `scripts/eval/`, `scripts/utils/`). 

The audit eliminated 5 critical (P0) bug classes and secondary (P1/P2) hygiene risks that previously rendered paper numbers vulnerable to silent fallbacks, unhandled crashes, or metric corruption:
1. **Unclear Exception Clauses & Silent Fallbacks**: Replaced bare `except:` and unqualified `except Exception: pass` in dispatch/measurement paths with narrow exception types (`(TypeError, ValueError)`, `(RuntimeError, torch.cuda.OutOfMemoryError)`) and structured logging (`logger.warning(..., exc_info=True)`).
2. **Attempt to Call Non-Callable Objects**: Added `assert callable(...)` and `isinstance(..., Callable)` guards before operator and compressor invocations, preventing crashes when registries return uninitialized instances or metadata.
3. **Unbound Local Variables & Closure Capture**: Refactored inner measurement callbacks (`fwd_bwd_step`, `comp_forward`, `run_step`) into parameter-bound factory functions (`_make_step`, `_make_forward`), eliminating Python closure variable deletion (`del mod, x`) that triggered unbound variable errors.
4. **Function-as-Value Timing & Synchronization**: Audited all CUDA event records and device synchronizations to ensure strict pre- and post-measurement barriers with accurate `start_event.elapsed_time(end_event)` or `time.perf_counter()` resolution.
5. **Outer Name & Loop Variable Shadowing**: Renamed inner loop indices (e.g. `b` $\rightarrow$ `b_cand`, `b_idx`; `h` $\rightarrow$ `head_kw`) preventing overwrites of outer tensor dimensions `B` (batch size) and `H` (head count).
6. **Robust Number Formatting**: Implemented `fmt_opt` helpers across reporting modules to eliminate crashes when formatting `None`, `NaN`, or string values.
7. **Entrypoint Standard Catalog**: Created canonical root CLI entrypoint `benchmarks/benchmark_all_108_cuda_isolated.py` and ensured uniform `# Author: Taha Majlesi` headers on all benchmark files.

---

## 2. Before vs. After Audit Matrix

| File Path | Bug Class / Issue | Before State | After State (Fix) | Impact on Trustworthiness |
|---|---|---|---|---|
| `scripts/analysis/export_master_matrix_latex.py` | P0: Bare Except (Bug 1) | Bare `except:` on line 27 silently catching all errors | Narrowed to `except (ValueError, TypeError):` | Prevents masking syntax/runtime errors during LaTeX matrix generation |
| `scripts/utils/nb_gen_10_12.py` | P0: Bare Except (Bug 1) | Bare `except:` on line 701 | Narrowed to `except (KeyError, ValueError, TypeError):` | Protects notebook generation from silent corruption |
| `scripts/benchmarks/run_paper_benchmarks.py` | P0: Unbound Variables (Bug 3) & Unquoted Keys | Unquoted dictionary keys `ppl_results[baseline][perplexity]` causing `NameError` | Quoted string access `ppl_results.get('baseline', {})...` + relative path resolution | Eliminates immediate crash during publication LaTeX table export |
| `scripts/benchmarks/run_qwen_llama_benchmarks.py` | P0: Syntax / Undefined Names (Bug 3, 8) | `mech.replace(_, )` syntax error; unquoted `Latency_ms`, `Peak_VRAM_MB` | Fixed string replacement `mech.replace('_', ' ')`, quoted DataFrame columns | Restores execution of Qwen-2.5 and LLaMA-3 surgery benchmarks |
| `scripts/benchmarks/benchmark_all_131_attentions.py` | P0: Unbound Variables in Closures (Bug 3) | `del mod, x` inside loop referencing closure `fwd_bwd_step()` | Bound via `_make_step(mod, x)` and reset via `mod = None; x = None` | Guarantees clean autograd execution and pyflakes clean pass |
| `scripts/benchmarks/benchmark_compression_pareto_master.py` | P0: Unbound Variables (Bug 3) & None formatting (Bug 8) | `del comp_layer, probe_x` causing closure unbound error; unhandled `None` in table | Bound via `_make_forward(...)`, assigned `None`, added `fmt_opt` | Prevents crash during 72-engine Pareto sweeps |
| `scripts/benchmarks/benchmark_long_context_scaling.py` | P0: Undefined Variable (Bug 3) & Closure leak | `FALLBACK_MAP` referenced but undefined; `del x` inside loop | Defined `FALLBACK_MAP` mapping; bound via `_make_step(model, x)` | Enables full 32k context extrapolation benchmarking |
| `scripts/benchmarks/benchmark_multi_precision_roofline.py` | P0: Unbound Variables (Bug 3) & None formatting (Bug 8) | `del x, mod` in loop; raw float formatting crashing on failed precisions | Bound via `_make_forward(...)`, added `fmt_opt` helper | Prevents crash when precision unsupported by hardware |
| `scripts/benchmarks/run_fla_head_to_head_benchmark.py` | P0: Non-Callable Call (Bug 2) & Broad Except (Bug 1) | `cls = ATTENTION_REGISTRY.get(...)` called without verification; broad `except Exception:` | Added `assert callable(cls)`; narrowed fallback to `except TypeError:` | Prevents `NoneType` call crashes and silent CUDA error masking |
| `scripts/benchmarks/run_new_triton_and_step_benchmark.py` | P0: Non-Callable Call (Bug 2) & Broad Except (Bug 1) | `m.step()` called without checking `hasattr/callable`; broad `except Exception:` | Added `callable(cls)`, `callable(m.step)` guards; narrowed to `(TypeError, ValueError)` | Protects autoregressive recurrent throughput benchmark |
| `scripts/benchmarks/batch_validate_all_compressors.py` | P0: Shadowing Outer Names (Bug 5) & Silent Except (Bug 1) | Loop variable `h` shadowed outer head/dim; silent `except Exception: pass` | Renamed to `head_kw`; added `logger.warning(..., exc_info=True)` | Prevents parameter freezing misconfiguration |
| `scripts/benchmarks/profile_peak_batch_saturation.py` | P0: Shadowing Outer Names (Bug 5) & Silent Break (Bug 1) | Loop variable `b` shadowed outer batch dimension; silent `except Exception: break` | Renamed to `b_cand`; added logging for unexpected non-OOM exceptions | Corrects binary search for peak batch saturation boundaries |
| `scripts/benchmarks/benchmark_all_108_cuda_modules.py` | P0: Silent CUDA Exception (Bug 1) | Silent `except Exception: pass` during reference execution | Replaced with `logger.warning(..., exc_info=True)` | Surfaced any reference evaluation failures |
| `scripts/benchmarks/benchmark_all_72_compressors.py` | P0: Non-Callable Target Guard (Bug 2) | Directly invoked `target(...)` from registry without callable assertion | Added `assert callable(target)` guard | Prevents non-callable registry artifact crashes |
| `scripts/benchmarks/benchmark_autograd_gradcheck_audit.py` | P0: Broad Exception Catching (Bug 1) | Broad `except Exception: pass` during module instantiation | Narrowed to `(TypeError, ValueError)` | Surfaced internal autograd failures while allowing shape fallbacks |
| `scripts/benchmarks/benchmark_48_triton_modules.py` | P0/P1: Broad Except (Bug 1) & KeyError/None format (Bug 8) | `except Exception:` on instantiation; `d2048.get(...):.4f` crashing on failed modules | Narrowed to `(TypeError, ValueError)`; formatted with `fmt_opt` | Prevents markdown table crash on partial benchmark runs |
| `scripts/benchmarks/build_gpu_cross_comparison.py` | P1: String formatting crash on None/NaN (Bug 8) | Unprotected `f"{c['taha_acc']:.2f}%"` crashing if metric is `None` or string | Formatted using `fmt_opt` helper with fallback to `"N/A"` | Guaranteed multi-GPU cross-comparison table generation |
| `benchmarks/*.py` (all 12 files) | Metadata Hygiene | Missing `# Author: Taha Majlesi` header | Added uniform `# Author: Taha Majlesi` header to all files | Conforms strictly to codebase authorship standards |
| `benchmarks/benchmark_all_108_cuda_isolated.py` | Suite Architecture Standard | Missing root entrypoint for isolated CUDA 108 benchmark | Created canonical wrapper delegating to `scripts/benchmarks/` | Standardizes CLI execution across root and scripts |

---

## 3. Verification & Compliance Evidence

### 3.1. Bare Except Audit
Command: `grep -rn -E "except\s*:" benchmarks/ scripts/`  
**Result**: `0 occurrences found` (PASSED).

### 3.2. Function-as-Value Timing Assignment Audit
Command: `grep -rn "=\s*torch\.cuda\.synchronize" benchmarks/ scripts/`  
**Result**: `0 occurrences found` (PASSED).

### 3.3. Authorship Header Audit
Command: `find benchmarks scripts -name "*.py" -exec grep -L "Taha Majlesi" {} +`  
**Result**: `0 files missing header` (PASSED across all 110+ files).

### 3.4. Pyflakes Static Analysis Verification
Executed on remote server `/mnt/d/SVD_linear_Attention`:
- **Undefined names**: `0`
- **Unbound variables**: `0`
- **Syntax errors**: `0`

### 3.5. Isolated Benchmark Smoke Test
Command: `python benchmarks/benchmark_all_108_cuda_isolated.py --modules 3` on remote GPU server.  
**Result**: Execution completed with isolated subprocess isolation verified.
