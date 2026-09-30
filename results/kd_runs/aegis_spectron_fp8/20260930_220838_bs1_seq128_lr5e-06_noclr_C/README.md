# KD Run: aegis_spectron_fp8

**Date:** 2026-09-30 22:24:53
**Module:** `aegis_spectron_fp8`
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

- Pre-KD PPL: 407.77
- Post-KD PPL: 83.79
- Recovery: 86.11%
- Post/Baseline ratio: 2.657x
- Step time: 1819 ms
- Peak VRAM: 1898 MB
