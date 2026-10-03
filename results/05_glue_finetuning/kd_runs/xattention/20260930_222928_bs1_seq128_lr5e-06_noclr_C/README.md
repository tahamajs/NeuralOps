# KD Run: xattention

**Date:** 2026-09-30 22:34:45
**Module:** `xattention`
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

- Pre-KD PPL: 47.62
- Post-KD PPL: 36.68
- Recovery: 68.03%
- Post/Baseline ratio: 1.163x
- Step time: 580 ms
- Peak VRAM: 1461 MB
