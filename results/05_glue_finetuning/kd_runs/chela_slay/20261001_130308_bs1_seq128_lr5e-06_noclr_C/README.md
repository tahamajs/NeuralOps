# KD Run: chela_slay

**Date:** 2026-10-01 13:38:44
**Module:** `chela_slay`
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

- Pre-KD PPL: 3368.59
- Post-KD PPL: 1346.73
- Recovery: 60.59%
- Post/Baseline ratio: 42.699x
- Step time: 2184 ms
- Peak VRAM: 2212 MB
