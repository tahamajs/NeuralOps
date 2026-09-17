# Skill — decay_linear_attention

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`decay_linear_attention` module. It covers everything specific to this module that the
generic skills do not.

## What is `decay_linear_attention`?

Decay Linear Attention: Recurrent Attention with Exponential Gating — implemented in `Codes/framework/attentions/decay_linear_attention.py`.

Mathematical core:
$$
S_t = \lambda_t S_{t-1} + K_t^\top V_t, \quad O_t = \frac{\phi(Q_t) S_t}{\phi(Q_t) \sum \lambda_t K_t^\top}
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`None (Full replacement architecture, lora_rank=16)` — see `04-rank-arg-mapping.md`.

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
