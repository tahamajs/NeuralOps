# SURGERY_prune_layers_GATES.md

## Gate 1: Works on ≥ 2 Model Families
- **Model Families**: `llama` (`model.layers`), `gpt2` (`transformer.h`), `qwen`
- **Evidence**: Verified via `_find_decoder_layers` and unit tests in `tests/surgery/test_prune_layers.py`.

## Gate 2: Works on ≥ 2 Model Sizes
- **Model Sizes**: Tested on toy models (6-layer, 32-dim) and standard HuggingFace depth configurations.
- **Evidence**: Unit test suite executes on multiple depth configurations.

## Gate 3: Reproducible Benchmark
- **Script**: `scripts/benchmarks/benchmark_prune_layers.py`

## Gate 4: Model Card on HF Hub
- **Hub Repository**: `https://huggingface.co/Hooshaai/neuralops-recipes`

## Gate 5: Recipe in neuralops-recipes Dataset
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **External User**: No external community verification logged yet.
- **Classification**: **EXPERIMENTAL** (`_experimental=True` enabled with runtime UserWarning).
