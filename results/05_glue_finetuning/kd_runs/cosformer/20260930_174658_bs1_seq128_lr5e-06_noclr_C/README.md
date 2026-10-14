# KD Run: cosformer

**Date:** 2026-09-30 17:48:17
**Module:** `cosformer`
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

- Pre-KD PPL: nan
- Post-KD PPL: nan
- Recovery: 0.00%
- Post/Baseline ratio: inf
- Step time: 594 ms
- Peak VRAM: 1207 MB
