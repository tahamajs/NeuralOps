# KD Run: orion

**Date:** 2026-10-01 11:39:18
**Module:** `orion`
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

- Pre-KD PPL: 451.57
- Post-KD PPL: 250.08
- Recovery: 47.97%
- Post/Baseline ratio: 7.929x
- Step time: 2120 ms
- Peak VRAM: 1153 MB
