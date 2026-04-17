# IITx-System-2 Attention Benchmark

Synthetic benchmark pack implementing tasks `I1`-`I12` from the IITx-System-2 brief.

## Layout

```text
iitx_system2_attention/
├── raw/
├── processed/
└── metadata/
```

## Generate

```bash
python /Users/tahamajs/Documents/uni/Research/consciousness/scripts/prepare_iitx_system2_attention_data.py --profile smoke --force
```

Available profiles:

- `smoke`: fast local generation.
- `v1`: larger recommendation-aligned split sizes (`I1~5k`, `I5~6k`, `I6~2k`).

## Evaluate

Prediction file contract (CSV):

- `id` (required)
- `prediction` (required)
- `confidence` (optional, float in `[0,1]`)
- `tokens_used` (optional, integer)

Run evaluator:

```bash
python /Users/tahamajs/Documents/uni/Research/consciousness/scripts/evaluate_iitx_system2_attention.py \
  --predictions /absolute/path/to/predictions.csv \
  --gold /Users/tahamajs/Documents/uni/Research/consciousness/data/benchmarks/reasoning/iitx_system2_attention/processed/iitx_system2_attention_test.csv \
  --output-json /Users/tahamajs/Documents/uni/Research/consciousness/reports/iitx_system2_attention_metrics.json \
  --output-csv /Users/tahamajs/Documents/uni/Research/consciousness/reports/iitx_system2_attention_metrics.csv
```

Strict mode (fails on missing/duplicate IDs):

```bash
python /Users/tahamajs/Documents/uni/Research/consciousness/scripts/evaluate_iitx_system2_attention.py \
  --predictions /absolute/path/to/predictions.csv \
  --strict
```

Metric groups exported:

- `M_alpha`: accuracy + ECE over `I1`, `I2`, `I9`, `I11`, `I12`
- `M_beta`: consistency metrics over `I3`, `I5`, `I8`
- `M_gamma`: token-efficiency metrics on `I4`
- `M_delta`: ablation and flip sensitivity metrics for `I6`, `I7`

## Output schema

Each row contains:

- task metadata: `task_id`, `task_sketch`, `metric_track`, `effort_tier`, `risk_tag`
- query/label contract: `prompt`, `expected_answer`, `answer_type`, `choices_json`
- pair controls: `pair_group`, `pair_variant`, `expected_behavior`
- robustness controls: `control_decoy`, `control_novelty`, `control_position`, `budget_tokens`
- task-specific structured details: `meta_json`

Pair semantics:

- `I5`, `I8`: `expected_behavior=invariant`
- `I7`: `expected_behavior=flip`
- `I6`: `expected_behavior=confidence_drop`
