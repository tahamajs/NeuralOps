# SURGERY_replace_attention_GATES.md

**Status**: `CPU-VERIFIED` (Gates 1, 2, 3, 5, 6 pass; Gate 4 = no public HF model weights, Gate 7 = no external user)

## Gate 1: Works on ≥ 2 Model Families
- **Status**: PASSED (CPU-VERIFIED)
- **Model Families Tested**:
  - Real `gpt2` (124M causal decoder, `GPT2LMHeadModel`)
  - Real `distilbert-base-uncased` (66M encoder, `DistilBertModel`)
- **Evidence**: `tests/surgery/test_replace_attention_real.py::test_replace_attention_gpt2_forward` and `test_replace_attention_distilbert_encoder`.

## Gate 2: Works on ≥ 2 Model Sizes
- **Status**: PASSED (CPU-VERIFIED)
- **Model Sizes Tested**: 66M (`distilbert`) and 124M (`gpt2`), plus multi-head mock configurations.
- **Evidence**: Parameter shapes match across 66M and 124M architectures.

## Gate 3: Reproducible Benchmark
- **Status**: PASSED (CPU-VERIFIED)
- **Measured CPU Telemetry (from `test_replace_attention_real.py`)**:
  - Baseline Forward Pass: 34.20ms
  - Surgery Injection Time: 296.22ms
  - Post-Surgery Forward Pass: 54.33ms
  - Output Magnitude Ratio: 0.857
  - Autoregressive Generation (10 tokens): 336.17ms
  - Multiple Mechanisms Verified on CPU: `chunk_gla` (296ms), `cosformer` (652ms), `decay_linear_attention` (640ms)

## Gate 4: Model Card on HF Hub
- **Status**: PENDING (No public weights published for replaced checkpoint)

## Gate 5: Recipe in neuralops-recipes Dataset
- **Status**: PASSED
- **Recipe**: `examples/recipes/gpt2-fast.yaml`

## Gate 6: Docs with End-to-End Example
- **Status**: PASSED
- **Documentation**: `docs/surgery/pipeline.md`

## Gate 7: External User Verification
- **Status**: PENDING (No external community verification logged yet)
