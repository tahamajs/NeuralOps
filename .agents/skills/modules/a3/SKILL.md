# Skill — a3

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`a3` module. It covers everything specific to this module that the
generic skills do not.

## What is `a3`?

A3: Activation-Aware SVD Compression with Fisher Energy Weighting — implemented in `Codes/framework/compression_engine/a3_compressor.py`.

Mathematical core:
$$
C_X = \frac{1}{N} \sum x_t x_t^\top, \quad F = \mathbb{E}[x^2], \quad W_{weighted} = W \operatorname{diag}(\sqrt{F+\epsilon}) = U\Sigma V^\top
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`rank / a3_rank (default 64), use_fisher (default True)` — see `04-rank-arg-mapping.md`.

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
