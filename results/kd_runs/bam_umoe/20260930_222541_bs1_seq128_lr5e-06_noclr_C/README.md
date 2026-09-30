# KD Run: bam_umoe

**Date:** 2026-09-30 22:30:08
**Module:** `bam_umoe`
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

- Pre-KD PPL: 173.00
- Post-KD PPL: 92.90
- Recovery: 56.62%
- Post/Baseline ratio: 2.946x
- Step time: 472 ms
- Peak VRAM: 1114 MB
