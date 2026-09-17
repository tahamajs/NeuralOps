# Forward Sanity Check — phantom

## Sanity Procedure

Check that softpick output probabilities sum to 1.0 and no tokens receive zero attention across all heads.

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
