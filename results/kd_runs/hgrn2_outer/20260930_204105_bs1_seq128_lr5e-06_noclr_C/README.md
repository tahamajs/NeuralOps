# KD Run: hgrn2_outer

**Date:** 2026-09-30 20:50:59
**Module:** `hgrn2_outer`
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

- Pre-KD PPL: 12352.75
- Post-KD PPL: 1014.88
- Recovery: 92.02%
- Post/Baseline ratio: 32.178x
- Step time: 1066 ms
- Peak VRAM: 1927 MB
