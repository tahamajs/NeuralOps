# KD Run: adasplash_attention

**Date:** 2026-09-30 22:08:02
**Module:** `adasplash_attention`
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

- Pre-KD PPL: 32.28
- Post-KD PPL: 31.80
- Recovery: 64.86%
- Post/Baseline ratio: 1.008x
- Step time: 1171 ms
- Peak VRAM: 1576 MB
