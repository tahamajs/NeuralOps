# Forward Sanity Check — svd_llm_v2

## Sanity Procedure

Validate that singular value sum cumulative energy exceeds energy_threshold (0.999) and output norm is non-zero.

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
