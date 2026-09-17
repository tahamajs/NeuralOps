# Skill — compressed_attention

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`compressed_attention` module. It covers everything specific to this module that the
generic skills do not.

## What is `compressed_attention`?

<FULL_NAME> — <one-paragraph description>

Mathematical core:
$$
<equation>
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`<rank_arg_name>` — see `04-rank-arg-mapping.md`.

## Special handling required

- `01-copy-weights-audit.md` — verify weight transfer
- `02-nested-encoder-check.md` — flat vs nested detection
- `03-causal-vs-bidirectional.md` — applicability to encoders
- `05-parameter-count-verify.md` — actual parameter change

## Do NOT

- Do NOT apply this module to <incompatible_backbone>
- Do NOT use rank > <max_rank>
- Do NOT skip the sanity check in `06-forward-sanity-check.md`

## Verification

Every run must pass `06-forward-sanity-check.md` before recovery fine-tuning
begins. If it fails, abort and log.
