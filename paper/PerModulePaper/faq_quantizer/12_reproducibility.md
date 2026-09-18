# 12. Reproducibility Guide

All source code, configuration artifacts, and empirical evaluation harnesses are organized to support deterministic end-to-end reproducibility across supported foundation architectures.

## 12.1 Hardware and Environment Setup
All experiments require Python 3.10+ and PyTorch 2.4+ configured with CUDA 12.1. Deterministic seeds are fixed across random number generators:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```

## 12.2 Unified Benchmark Sweeps
To reproduce the empirical benchmark evaluations for `faq_quantizer` across GLUE classification tasks:

```bash
# 1. Run unified compression / attention sweep on DistilBERT
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor faq_quantizer \
    --dataset sst2 \
    --batch_size 64 \
    --recovery_steps 50 \
    --output_dir results/unified_faq_quantizer/distilbert

# 2. Execute multi-model verification suite across backbones
python -u scripts/run_all_4_faq_quantizer.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --batch_size 64 \
    --output_dir results/unified_faq_quantizer
```

## 12.3 Automated Quality Gate Verification
Verification of generated model weights, peak memory envelopes, and downstream accuracy preservation against pre-set tolerances is executed via:
```bash
python3 scripts/verify_module_paper.py --module faq_quantizer --strict
```
Execution logs, tensor shapes, and per-step latency metrics are saved automatically to `results/unified_faq_quantizer/` for inspection and audit.
