# Forward Sanity Check — decay_linear_attention

## Sanity Procedure

Verify linear recurrence running sum does not produce NaNs and denominator is clamped to >= 1e-5.

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
