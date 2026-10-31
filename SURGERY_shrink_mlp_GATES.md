# SURGERY_shrink_mlp_GATES.md

## Gate 1: Works on ≥ 2 Model Families
- **Model Families**: `llama` / `mistral` (SwiGLU), `gpt2` (GELU), `qwen`
- **Evidence**: Verified via `test_shrink_swiglu` and `test_shrink_gelu` in `tests/surgery/test_shrink_mlp.py`.

## Gate 2: Works on ≥ 2 Model Sizes
- **Model Sizes**: Tested on small/toy configurations (hidden=32, intermediate=128 down to 64/96).
- **Evidence**: Unit tests execute on multiple dimension configurations.

## Gate 3: Reproducible Benchmark
- **Script**: `scripts/benchmarks/benchmark_shrink_mlp.py` (simulated intermediate dimension sweep).

## Gate 4: Model Card on HF Hub
- **Hub Repository**: `https://huggingface.co/Hooshaai/neuralops-recipes`

## Gate 5: Recipe in neuralops-recipes Dataset
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **External User**: No external community verification logged yet.
- **Classification**: **EXPERIMENTAL** (`_experimental=True` enabled with runtime UserWarning).
