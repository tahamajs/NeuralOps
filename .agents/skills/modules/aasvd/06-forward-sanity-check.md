# Forward Sanity Check — aasvd

## Sanity Procedure

Forward pass check on SVDLinear ensuring output std > 0.01 and non-NaN values.

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
