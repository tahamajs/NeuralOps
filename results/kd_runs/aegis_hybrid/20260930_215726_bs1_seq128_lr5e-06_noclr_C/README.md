# KD Run: aegis_hybrid

**Date:** 2026-09-30 22:03:59
**Module:** `aegis_hybrid`
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

- Pre-KD PPL: 371.97
- Post-KD PPL: 431.19
- Recovery: -17.40%
- Post/Baseline ratio: 13.671x
- Step time: 707 ms
- Peak VRAM: 1182 MB
