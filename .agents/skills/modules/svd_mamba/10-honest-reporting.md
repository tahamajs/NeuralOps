# Honest Reporting — svd_mamba

## Purpose

Decide how to report this module's results truthfully.

## Decision matrix

| Actual situation | How to report |
|---|---|
| Params decreased, accuracy improved | "compression" |
| Params decreased, accuracy kept | "lossless compression" |
| Params decreased, accuracy dropped | "compression with X% drop" |
| Params unchanged, accuracy kept | "re-architecture" |
| Params increased, accuracy kept | "re-architecture, higher capacity" |
| Params increased, accuracy dropped | "negative result" |
| OOM on some backbones | "memory-limited, works on N of 3" |
| 0.00% on all backbones | "incompatible architecture — see failure signature" |

## Required disclosures in abstract

- If params increased: state the increase
- If latency increased: state the factor
- If accuracy dropped: state the delta
- If some backbones failed: name them
- If single-seed: say so

## Anti-patterns to avoid

- "significant improvement" without p-value
- "state-of-the-art" without comparison
- Hiding failed runs
- Reporting only the best seed
- Using "compression" when params increased

## Output

`papers/modules/svd_mamba/metrics/honest_reporting.json`
