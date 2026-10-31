# SURGERY_distill_GATES.md

**Status**: `STUB` (Documented only; contract verified)

## Gate 1: Works on ≥ 2 Model Families
- **Status**: STUB / NOT IMPLEMENTED
- **Evidence**: `tests/surgery/test_distill.py` and `tests/surgery/test_stub_honesty.py`

## Gate 2: Works on ≥ 2 Model Sizes
- **Status**: STUB

## Gate 3: Reproducible Benchmark
- **Status**: FAILED / BLOCKED
- **Blocker**: Multi-pillar gradient distillation requires CUDA GPU hardware for student-teacher backpropagation. Not functional on CPU.

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
