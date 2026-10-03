# KD Run: cosformer

**Date:** 2026-10-01 00:23:44
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
- Post-KD PPL: 5058.89
- Recovery: 87.24%
- Post/Baseline ratio: 160.396x
- Step time: 3230 ms
- Peak VRAM: 1911 MB
