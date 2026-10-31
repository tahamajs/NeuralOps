# SURGERY_shrink_mlp_GATES.md

**Status**: `CPU-VERIFIED`

## Gate 1: Works on ≥ 2 Model Families
- **Status**: PASSED (CPU-VERIFIED)
- **Model Families Tested**:
  - Real `gpt2` (GELU via `Conv1D`)
  - Real `distilbert-base-uncased` (GELU via `nn.Linear` `lin1`/`lin2`)
  - Mock SwiGLU / LLaMA (`gate_proj`, `up_proj`, `down_proj`)
- **Evidence**: `tests/surgery/test_shrink_mlp_real.py`

## Gate 2: Works on ≥ 2 Model Sizes
- **Status**: PASSED (CPU-VERIFIED)
- **Model Sizes Tested**: 66M (`distilbert`) and 124M (`gpt2`).

## Gate 3: Reproducible Benchmark
- **Status**: PASSED (CPU-VERIFIED)
- **Measured CPU Telemetry (from `test_shrink_mlp_real.py`)**:
  - GPT-2 (124M): 124,439,808 -> 113,096,748 parameters (9.12% reduction at ratio=0.80) in 750.01ms
  - Logits Cosine Similarity: 0.9414 (preserves representation intent)
  - Perplexity Retention (ratio=0.90): PPL before = 40.56, PPL after = 44.53 (1.10x factor, within 20% budget)
  - DistilBERT (66M): 66,362,880 -> 60,691,350 parameters (8.55% reduction at ratio=0.80) in 85.20ms

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
