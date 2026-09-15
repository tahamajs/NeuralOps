# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Very Low Rank (r=8)

**Result.** {{R8_ACC}}%. DCT truncation loses too much high-frequency info.

### D.1.2 Learned Basis

**Goal.** Replace DCT with learned projection.

**Result.** No improvement, more parameters. DCT is already optimal for 
smooth attention.

### D.1.3 Applying Galerkin to Decoders

**Result.** Causality broken. Output incoherent.

### D.1.4 No Energy Scaling

**Result.** {{NO_SCALE_ACC}}%. Output magnitudes drift.

### D.1.5 Very High Rank (r=256)

**Result.** {{R256_ACC}}%. No accuracy benefit, slower.

### D.1.6 No QR Orthonormalization

**Result.** Accumulating floating-point drift; slight accuracy drop.

## D.2 Hyperparameter Failures

| LR | D acc |
|---|---|
| 5e-5 | {{D_LR5e5}} |
| 1e-4 | {{D_LR1e4}} |
| 2e-4 | {{D_ACC}} |
| 5e-4 | {{D_LR5e4}} |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | {{SEED42_D}} | {{SEED42_R}} | {{SEED42_G}} |
| 123 | {{SEED123_D}} | {{SEED123_R}} | {{SEED123_G}} |
| 456 | {{SEED456_D}} | {{SEED456_R}} | {{SEED456_G}} |
| Mean | {{MEAN_D}} | {{MEAN_R}} | {{MEAN_G}} |
| Std | {{STD_D}} | {{STD_R}} | {{STD_G}} |

## D.4 Lessons Learned

1. Encoder-only benefit is real.
2. Decoder fallback necessary.
3. DCT beats learned basis.
4. Rank 64 is a sweet spot.
5. Energy scaling is essential.

## D.5 Unresolved Questions

1. **Causal Galerkin** — is it possible?
2. **Learned basis** — could it help for specific tasks?
3. **Very long context** ($N > 4096$) — does DCT scaling hold?
