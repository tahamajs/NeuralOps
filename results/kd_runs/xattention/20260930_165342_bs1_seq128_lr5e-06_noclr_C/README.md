# KD Run: xattention

**Date:** 2026-09-30 17:04:03
**Module:** `xattention`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 128
- Recovery steps: 1000
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 47.62
- Post-KD PPL: 36.10
- Recovery: 71.64%
- Post/Baseline ratio: 1.145x
- Step time: 559 ms
- Peak VRAM: 1460 MB
