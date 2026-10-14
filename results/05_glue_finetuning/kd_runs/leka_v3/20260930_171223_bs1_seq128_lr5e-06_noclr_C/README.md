# KD Run: leka_v3

**Date:** 2026-09-30 17:19:44
**Module:** `leka_v3`
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

- Pre-KD PPL: 711.96
- Post-KD PPL: 261.47
- Recovery: 66.21%
- Post/Baseline ratio: 8.290x
- Step time: 792 ms
- Peak VRAM: 1235 MB
