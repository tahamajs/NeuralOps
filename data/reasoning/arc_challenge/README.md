# ARC-Challenge Benchmark Package

This directory hosts the local data package for the `arc_challenge` benchmark
used by the IIT-AM benchmark runner.

## Layout

```text
arc_challenge/
├── raw/
│   └── .gitkeep
├── processed/
│   ├── arc_challenge_train.csv
│   ├── arc_challenge_validation.csv
│   └── arc_challenge_test.csv
└── metadata/
    └── manifest.json
```

## Notes

- `processed/` currently includes a compact smoke subset so benchmark
  orchestration can run end-to-end without external download requirements.
- For larger local packs, place raw JSONL dumps into `raw/` and run:

```bash
python /Users/tahamajs/Documents/uni/Research/consciousness/scripts/prepare_arc_challenge_data.py
```

## Expected Processed Schema

Each CSV row must provide:

- `id`
- `split`
- `question`
- `choice_a`
- `choice_b`
- `choice_c`
- `choice_d`
- `answer_label`
- `answer_text`
- `source`
- `notes`
