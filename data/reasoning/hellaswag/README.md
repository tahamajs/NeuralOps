# HellaSwag Benchmark Package

This package contains a local reproducible subset for `hellaswag`.

## Layout

```text
hellaswag/
├── raw/
├── processed/
│   ├── hellaswag_train.csv
│   ├── hellaswag_validation.csv
│   └── hellaswag_test.csv
└── metadata/
    └── manifest.json
```

The processed CSV schema is:

- `id, split, question, choice_a, choice_b, choice_c, choice_d, answer_label, answer_text, source, notes`
