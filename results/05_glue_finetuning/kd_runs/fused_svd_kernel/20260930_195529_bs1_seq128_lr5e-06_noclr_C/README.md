# KD Run: fused_svd_kernel

**Date:** 2026-09-30 20:48:36
**Module:** `fused_svd_kernel`
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

- Pre-KD PPL: 339.36
- Post-KD PPL: 281.41
- Recovery: 18.83%
- Post/Baseline ratio: 8.922x
- Step time: 5790 ms
- Peak VRAM: 1053 MB
