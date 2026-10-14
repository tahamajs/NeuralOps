# KD Run: aegis_v6

**Date:** 2026-09-30 20:04:19
**Module:** `aegis_v6`
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

- Pre-KD PPL: 156.83
- Post-KD PPL: 156.83
- Recovery: 0.00%
- Post/Baseline ratio: 4.972x
- Step time: 954 ms
- Peak VRAM: 1470 MB
