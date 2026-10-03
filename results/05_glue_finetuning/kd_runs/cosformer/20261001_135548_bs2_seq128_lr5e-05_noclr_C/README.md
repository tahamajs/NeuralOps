# KD Run: cosformer

**Date:** 2026-10-01 14:19:28
**Module:** `cosformer`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 2
- Grad accumulation: 4 (effective batch = 8)
- Sequence length: 128
- Recovery steps: 500
- Learning rate: 5.00e-05
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 39435.26
- Post-KD PPL: 2532.71
- Recovery: 93.65%
- Post/Baseline ratio: 80.302x
- Step time: 2770 ms
- Peak VRAM: 1911 MB
