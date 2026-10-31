# SURGERY_quantize_int4_GATES.md

**Status**: `STUB` (Documented only; fail-closed contract verified)

## Gate 1: Works on ≥ 2 Model Families
- **Status**: STUB / NOT IMPLEMENTED
- **Evidence**: `tests/surgery/test_stub_honesty.py::test_quantize_int4_fail_closed_and_validation`

## Gate 2: Works on ≥ 2 Model Sizes
- **Status**: STUB

## Gate 3: Reproducible Benchmark
- **Status**: FAILED / BLOCKED
- **Blocker**: Requires external quantization engine (`autoawq` or `gptqmodel`) and CUDA GPU hardware for calibration profiling. Not functional on CPU.

## Gate 4: Model Card on HF Hub
- **Status**: PENDING

## Gate 5: Recipe in neuralops-recipes Dataset
- **Status**: PASSED (Schema spec only)
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Status**: PASSED
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **Status**: PENDING
