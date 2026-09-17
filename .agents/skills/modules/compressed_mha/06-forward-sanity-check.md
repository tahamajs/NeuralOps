# Forward Sanity Check — compressed_mha

## Sanity Procedure

Verify forward pass output shape matches [B, S, D] and relative reconstruction error rel_err < 0.30.

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
