# Skill — zerokv_neo

## Purpose

This skill guides an agent through testing, validating, and reporting on the
`zerokv_neo` module. It covers everything specific to this module that the
generic skills do not.

## What is `zerokv_neo`?

ZeroKV-Neo: Outlier-Preserving KV Cache Pruning with Attention Sinks — implemented in `Codes/framework/compression_engine/zerokv_neo_compressor.py`.

Mathematical core:
$$
S = \mathrm{mean}_h(A_{h, :, :}), \quad \mathcal{I}_{sink} = \{0, \dots, s-1\}, \quad \mathcal{I}_{top} = \operatorname{topk}(S, k), \quad \mathcal{K} = \mathcal{I}_{sink} \cup \mathcal{I}_{top}
$$

## Known failure modes

Listed in `09-failure-signature.md`. Every agent working with this module must
read that file first.

## Rank argument

`keep_ratio (default 0.3), num_attention_sinks (default 4)` — see `04-rank-arg-mapping.md`.

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
