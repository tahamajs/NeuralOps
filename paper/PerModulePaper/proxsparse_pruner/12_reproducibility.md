# 12. Reproducibility Guide

To reproduce the empirical benchmark evaluations for `proxsparse_pruner`:

```bash
# 1. Run unified compression / attention sweep
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor proxsparse_pruner \
    --dataset sst2 \
    --batch_size 64 \
    --recovery_steps 50 \
    --output_dir results/unified_proxsparse_pruner

# 2. Execute GPU matrix verification
python -u Codes/framework/run_quality_sweep.py \
    --model roberta-base \
    --model_family roberta \
    --dataset sst2 \
    --rank 16 \
    --batch_size 64 \
    --output_dir results/unified_proxsparse_pruner/roberta
```
