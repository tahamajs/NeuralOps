# KD Run: eat

**Date:** 2026-09-30 22:51:54
**Module:** `eat`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 96
- Recovery steps: 300
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 34.01
- Post-KD PPL: 34.55
- Recovery: 0.00%
- Post/Baseline ratio: 1.016x
- Step time: 275 ms
- Peak VRAM: 1376 MB
