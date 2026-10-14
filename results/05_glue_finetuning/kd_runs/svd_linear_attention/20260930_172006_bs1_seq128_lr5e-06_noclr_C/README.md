# KD Run: svd_linear_attention

**Date:** 2026-09-30 17:23:53
**Module:** `svd_linear_attention`
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

- Pre-KD PPL: 339.37
- Post-KD PPL: 174.36
- Recovery: 53.60%
- Post/Baseline ratio: 5.528x
- Step time: 363 ms
- Peak VRAM: 1064 MB
