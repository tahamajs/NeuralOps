# KD Run: kda_attention

**Date:** 2026-10-01 00:32:02
**Module:** `kda_attention`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 2
- Grad accumulation: 4 (effective batch = 8)
- Sequence length: 128
- Recovery steps: 1000
- Learning rate: 2.00e-05
- KL temperature: 2.0
- LR schedule: cosine
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 2796.78
- Post-KD PPL: 77.16
- Recovery: 98.35%
- Post/Baseline ratio: 2.446x
- Step time: 1557 ms
- Peak VRAM: 1957 MB
