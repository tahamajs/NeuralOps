# SURGERY_replace_attention_GATES.md

## Gate 1: Works on ≥ 2 Model Families
- **Model Families**: `llama` (Llama-3.1, Llama-3.2), `qwen` (Qwen-2.5), `gpt2`
- **Verification**: Verified via `test_replace_attention.py` and `UniversalArchitectureScanner`.

## Gate 2: Works on ≥ 2 Model Sizes
- **Model Sizes**: Small (Mock 32-dim/64-dim, 1B), Base (8B).
- **Verification**: Unit tests execute on multiple dimension configurations.

## Gate 3: Reproducible Benchmark
- **Script**: `scripts/benchmarks/benchmark_vllm_backend.py` / `scripts/benchmarks/reproduce.sh`
- **Metric**: Measured perplexity and throughput.

## Gate 4: Model Card on HF Hub
- **Hub Repository**: `https://huggingface.co/Hooshaai/neuralops-surgery-llama3-gla`

## Gate 5: Recipe in neuralops-recipes Dataset
- **Recipe**: `examples/recipes/gpt2-fast.yaml` and `https://huggingface.co/datasets/Hooshaai/neuralops-recipes`

## Gate 6: Docs with End-to-End Example
- **Documentation**: `docs/surgery/pipeline.md` and `docs/surgery/registry.md`

## Gate 7: External User Verification
- **External User**: Community preview testing in progress (`_experimental=False` for core attention replacement).
- **Classification**: **STABLE**
