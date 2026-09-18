# 12. Reproducibility Guide

To reproduce the empirical benchmark evaluations for `block_transformer_hiba`:

```bash
# 1. Run unified compression / attention sweep
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor block_transformer_hiba \
    --dataset sst2 \
    --batch_size 64 \
    --recovery_steps 50 \
    --output_dir results/unified_block_transformer_hiba

# 2. Execute GPU matrix verification
python -u Codes/framework/run_quality_sweep.py \
    --model roberta-base \
    --model_family roberta \
    --dataset sst2 \
    --rank 16 \
    --batch_size 64 \
    --output_dir results/unified_block_transformer_hiba/roberta
```
