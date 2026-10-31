# SURGERY_distill_GATES.md

## Gate 1: Works on ≥ 2 Model Families
- **Model Families**: `llama`, `gpt2`, `qwen`, `mistral`
- **Evidence**: Verified with 6-pillar objective configurations in `tests/surgery/test_distill.py`.

## Gate 2: Works on ≥ 2 Model Sizes
- **Model Sizes**: Configured for lightweight recovery and full 8B distillation loops.
- **Evidence**: Pillar objectives tested with variable dimension scaling.

## Gate 3: Reproducible Benchmark
- **Script**: `scripts/benchmarks/benchmark_distill.py`

## Gate 4: Model Card on HF Hub
- **Hub Repository**: `https://huggingface.co/Hooshaai/neuralops-recipes`

## Gate 5: Recipe in neuralops-recipes Dataset
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **External User**: No external community verification logged yet.
- **Classification**: **EXPERIMENTAL** (`_experimental=True` enabled with runtime UserWarning, GPU distillation recommended).
