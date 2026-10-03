# KD Run: aa_svd

**Date:** 2026-09-30 18:09:47
**Module:** `aa_svd`
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

- Pre-KD PPL: 47.22
- Post-KD PPL: 40.11
- Recovery: 45.34%
- Post/Baseline ratio: 1.272x
- Step time: 245 ms
- Peak VRAM: 1334 MB
