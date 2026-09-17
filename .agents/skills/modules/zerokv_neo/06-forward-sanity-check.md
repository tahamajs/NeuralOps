# Forward Sanity Check — zerokv_neo

## Sanity Procedure

Pass test tensor [1, 16, 768] through model forward; ensure output logits have non-zero standard deviation and no NaNs.

- Pass synthetic input tokens [B=2, S=16] through model.
- Confirm non-zero standard deviation and absence of NaNs/Infs.
