# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `huffman_coder`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor huffman_coder \
    --output_dir results/unified_huffman_coder
```
