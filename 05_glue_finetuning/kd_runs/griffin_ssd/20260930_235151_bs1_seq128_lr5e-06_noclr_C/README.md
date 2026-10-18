# KD Run: griffin_ssd

**Date:** 2026-09-30 23:57:09
**Module:** `griffin_ssd`
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

- Pre-KD PPL: 413.30
- Post-KD PPL: 37.86
- Recovery: 98.34%
- Post/Baseline ratio: 1.200x
- Step time: 573 ms
- Peak VRAM: 1625 MB
