# KD Run: triton_svd_linear_attention

**Date:** 2026-10-01 14:12:37
**Module:** `triton_svd_linear_attention`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 128
- Recovery steps: 500
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 489.13
- Post-KD PPL: 226.84
- Recovery: 57.32%
- Post/Baseline ratio: 7.192x
- Step time: 404 ms
- Peak VRAM: 1150 MB
