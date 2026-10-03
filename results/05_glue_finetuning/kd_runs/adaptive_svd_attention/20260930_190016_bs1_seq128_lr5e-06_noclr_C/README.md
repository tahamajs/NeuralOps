# KD Run: adaptive_svd_attention

**Date:** 2026-09-30 19:04:20
**Module:** `adaptive_svd_attention`
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

- Pre-KD PPL: 339.43
- Post-KD PPL: 184.52
- Recovery: 50.31%
- Post/Baseline ratio: 5.850x
- Step time: 364 ms
- Peak VRAM: 1077 MB
