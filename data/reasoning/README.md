# Reasoning Benchmarks

This folder groups reasoning-oriented benchmark packages used by IIT-AM and
related experiments.

## Open-suite packages (runner-integrated)

- `arc_challenge/`
- `commonsenseqa/`
- `hellaswag/`

Each package follows:

```text
<benchmark>/
├── raw/
├── processed/
└── metadata/
```

## Other reasoning datasets

- `gsm8k_style/`: synthetic arithmetic chain-of-thought style benchmark.
- `arc_style/`: legacy synthetic ARC-style benchmark used by IIT-A workflow.
- `iitx_system2_attention/`: synthetic IITx-System-2 benchmark (`I1`-`I12`)
  with flip/invariance/ablation controls.

## Manifest refresh

```bash
python /Users/tahamajs/Documents/uni/Research/consciousness/scripts/prepare_open_reasoning_data.py --benchmark all
```
