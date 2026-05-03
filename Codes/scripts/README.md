# 🧪 Experiments & Scripts

This directory contains **end‑to‑end experiment runners**, **dataset scripts**, and **utility helpers** for the Efficient Transformer Toolkit.

## 📁 Structure

```
experiments/
├── run_experiment.py         # Core experiment driver (command‑line)
├── scheduler.py              # Grid search scheduler (calls run_experiment)
├── run_sweep.py              # Another parameter sweep variant
├── test_compressed_model.py  # Quick test of a compressed model
├── final_optimized_svd.py    # Self‑contained optimization pipeline
├── generate_report.py        # HTML/CSV report generator from results
├── generate_synthetic_data.py# Synthetic data generation
├── inspect_datasets.py       # Dataset structure inspector
├── export_coreml.py          # CoreML export (for SOA models)
├── download_model.py         # Fast model download with custom endpoint
├── create_extension.py       # Scaffold new attention/compressor modules
└── README.md
```

## 🚀 Quick Start

### Run a Single Experiment

```bash
python experiments/run_experiment.py \
    --model distilbert-base-uncased \
    --attention aurora \
    --rank 64 \
    --task mrpc \
    --epochs 5 \
    --batch-size 32
```

Results are automatically saved in `results/<task>_<attention>_<timestamp>/`.

### Run a Grid Search

Edit `scheduler.py` or `run_sweep.py` with your search space, then:

```bash
python experiments/run_sweep.py
```

### Generate a Report

```bash
python experiments/generate_report.py --results_dir results/mrpc_aurora_20260101_120000 --output_dir report
```

### Inspect Datasets

```bash
python experiments/inspect_datasets.py
```

## 🔧 Extension Templates

Create new modules quickly:

```bash
python experiments/create_extension.py --type attention --name my_attention
python experiments/create_extension.py --type compressor --name my_compressor
```

## 📦 Adding New Experiments

1. Copy `run_experiment.py` and modify.
2. Use the `ResultsManager` to log metrics and save outputs.
3. Add your script here.

---

*The experiments directory – from single runs to full sweeps.*