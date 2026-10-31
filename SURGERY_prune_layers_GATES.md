# SURGERY_prune_layers_GATES.md

**Status**: `PARTIAL` (Structural layer removal verified on CPU; healing requires GPU)

## Gate 1: Works on ≥ 2 Model Families
- **Status**: PASSED (Structural Layer Removal)
- **Model Families Tested**:
  - `gpt2` (`transformer.h`)
  - `distilbert` (`transformer.layer`)
  - `llama` (`model.layers`)
- **Evidence**: `tests/surgery/test_prune_layers.py` and `tests/surgery/test_surgery_exhaustive.py`

## Gate 2: Works on ≥ 2 Model Sizes
- **Status**: PASSED
- **Model Sizes Tested**: Depth truncation verified on 2-layer to 12-layer backbones.

## Gate 3: Reproducible Benchmark
- **Status**: PARTIAL
- **Note**: Structural layer removal works cleanly and updates `model.config.num_hidden_layers`. However, post-pruning representation healing requires multi-pillar distillation on GPU.

## Gate 4: Model Card on HF Hub
- **Status**: PENDING

## Gate 5: Recipe in neuralops-recipes Dataset
- **Status**: PASSED
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Status**: PASSED
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **Status**: PENDING
