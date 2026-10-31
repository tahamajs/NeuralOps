# SURGERY_quantize_int4_GATES.md

## Gate 1: Works on ≥ 2 Model Families
- **Model Families**: `llama` (Llama-3.1, Llama-3.2), `gpt2`, `qwen`
- **Evidence**: Verified via fail-closed architecture checks and method dispatching.

## Gate 2: Works on ≥ 2 Model Sizes
- **Model Sizes**: Configured for 4-bit quantization on sub-1B and 8B models.
- **Evidence**: Parameterized group size and bit precision contracts verified in tests.

## Gate 3: Reproducible Benchmark
- **Script**: `scripts/benchmarks/benchmark_quantize_int4.py`

## Gate 4: Model Card on HF Hub
- **Hub Repository**: `https://huggingface.co/Hooshaai/neuralops-recipes`

## Gate 5: Recipe in neuralops-recipes Dataset
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **External User**: No external community verification logged yet.
- **Classification**: **EXPERIMENTAL** (`_experimental=True` enabled with runtime UserWarning, requires GPU + calibration).
