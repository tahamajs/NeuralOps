# KD Run: factorised_matrix_state

**Date:** 2026-09-30 17:03:50
**Module:** `factorised_matrix_state`
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

- Pre-KD PPL: 3867.96
- Post-KD PPL: 114.52
- Recovery: 97.84%
- Post/Baseline ratio: 3.631x
- Step time: 419 ms
- Peak VRAM: 1754 MB
