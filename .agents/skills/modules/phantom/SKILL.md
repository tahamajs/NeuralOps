# Skill — phantom

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`phantom` module. It covers everything specific to this module that the
generic skills do not.

## What is `phantom`?

Phantom: Parameter-Hashed Attention with Token-wise Orthogonal Mapping — implemented in `Codes/framework/attentions/phantom.py`.

Mathematical core:
$$
W \approx C U^+ R, \quad A_{i,j} = \operatorname{Softpick}\left(\frac{Q K^\top}{\sqrt{d}} \odot M_{LSH}\right)
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`cur_rank (default 16), lsh_rounds (default 4)` — see `04-rank-arg-mapping.md`.

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
