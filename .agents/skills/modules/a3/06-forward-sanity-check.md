# Forward Sanity Check — a3

## Sanity Procedure

Verify that reconstructed weight W_rec = U @ V has relative error <= max_error (0.40).

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
