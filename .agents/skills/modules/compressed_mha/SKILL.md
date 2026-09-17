# Skill — compressed_mha

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`compressed_mha` module. It covers everything specific to this module that the
generic skills do not.

## What is `compressed_mha`?

Compressed MHA: Joint Truncated SVD Multi-Head Attention — implemented in `Codes/framework/attentions/compressed_mha_attention.py`.

Mathematical core:
$$
W_{qkv} = \begin{bmatrix} W_q \\ W_k \\ W_v \end{bmatrix} \approx U_r \Sigma_r V_r^\top, \quad W_o \approx U_{ro} \Sigma_{ro} V_{ro}^\top
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`rank_qkv (default 64), rank_out (default 64)` — see `04-rank-arg-mapping.md`.

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
