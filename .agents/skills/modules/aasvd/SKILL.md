# Skill — aasvd

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`aasvd` module. It covers everything specific to this module that the
generic skills do not.

## What is `aasvd`?

AASVD: Activation-Aware SVD Compressor with Streaming EMA Covariance — implemented in `Codes/framework/compression_engine/aa_svd_compressor.py`.

Mathematical core:
$$
C_X^{(t)} = \beta C_X^{(t-1)} + (1 - \beta) \frac{1}{B} X^\top X, \quad y = (x V^\top) U^\top + \frac{\alpha}{r} (x A^\top) B^\top
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`rank (default 288), ema_beta (default 0.9)` — see `04-rank-arg-mapping.md`.

## Special handling required

- `01-copy-weights-audit.md` — verify weight transfer
- `02-nested-encoder-check.md` — flat vs nested detection
- `03-causal-vs-bidirectional.md` — applicability to encoders
- `05-parameter-count-verify.md` — actual parameter change

## Do NOT

- Do NOT claim parameter compression if parameter change is positive or zero
- Do NOT skip the sanity check in `06-forward-sanity-check.md`
- Do NOT ignore the 50.92% classification collapse pattern

## Verification

Every run must pass `06-forward-sanity-check.md` before recovery fine-tuning
begins. If it fails, abort and log.
