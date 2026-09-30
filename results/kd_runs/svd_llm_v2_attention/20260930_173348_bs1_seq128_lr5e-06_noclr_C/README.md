# KD Run: svd_llm_v2_attention

**Date:** 2026-09-30 17:40:31
**Module:** `svd_llm_v2_attention`
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

- Pre-KD PPL: 339.46
- Post-KD PPL: 120.41
- Recovery: 71.14%
- Post/Baseline ratio: 3.818x
- Step time: 526 ms
- Peak VRAM: 1132 MB
