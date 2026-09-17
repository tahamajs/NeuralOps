# Skill — svd_llm_v2

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`svd_llm_v2` module. It covers everything specific to this module that the
generic skills do not.

## What is `svd_llm_v2`?

SVD-LLM V2 Attention: Truncation-Aware SVD Attention with LoRA Recovery — implemented in `Codes/framework/attentions/svd_llm_v2_attention.py`.

Mathematical core:
$$
W_q, W_k, W_v, W_o \approx U_r \Sigma_r V_r^\top, \quad \Delta W = \frac{\alpha}{r_{lora}} B A
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`rank (default 192), energy_threshold (default 0.999)` — see `04-rank-arg-mapping.md`.

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
