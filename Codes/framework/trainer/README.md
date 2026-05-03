# 🏋️ Training Module

The training module provides a **unified interface** for training Transformer models with the Efficient Transformer Toolkit.  
It supports **standard HuggingFace Trainer**, **custom training loops**, **Unsloth‑accelerated training**, and **Reinforcement Learning (via TRL)**.  
All runs are automatically saved into organized, timestamped directories under `results/`.

## 📁 Structure

```
framework/training/
├── __init__.py
├── trainer.py              # SVDAwareTrainer (with SVD‑specific callbacks)
├── callbacks.py            # Memory, Time, GradientNorm, SVDAttention callbacks
├── hyperopt.py             # Optuna‑based hyperparameter optimization
├── metrics.py              # Per‑GLUE‑task metric functions
├── utils.py                # Custom training loop, evaluation, benchmarks
├── results_manager.py      # Manages per‑run directories, logging, saving
├── trainer_factory.py      # Factory to select trainer (HF, Unsloth, RL)
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Using the Trainer Factory

```python
from framework.training import get_trainer, ResultsManager

config = {
    "experiment_name": "mrpc_aurora",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
}
trainer, results_mgr = get_trainer(
    model, train_dataset, eval_dataset,
    config=config,
    compute_metrics=compute_metrics_fn
)
trainer.train()
```

After training, you'll find everything in `results/mrpc_aurora_<timestamp>/`:
- `config.yaml`
- `metrics.jsonl`
- `training_history.png`
- `checkpoints/`
- `logs/`
- `final_model/`

### 2. Using a Custom Training Loop

```python
from framework.training.utils import train_model

results = train_model(model, train_loader, val_loader, device, config, num_labels)
```

### 3. With Unsloth

Set `use_unsloth=True` in the config passed to `get_trainer`. Unsloth must be installed.

### 4. With Reinforcement Learning (TRL)

Set `use_rl=True` and provide a `tokenizer`. TRL must be installed.

## 🧩 Components

### ResultsManager

Handles the creation of a unique run directory and saving of:
- **Configuration** (`config.yaml`)
- **Metrics** (`metrics.jsonl`, one JSON per line)
- **Training curves** (`training_history.png`)
- **Model checkpoints** (HuggingFace Trainer saves them; the manager stores the final model separately)
- **Logs** (console + file)

### Callbacks

- `MemoryCallback` – logs GPU/CPU memory usage
- `TimeCallback` – logs elapsed time
- `GradientNormCallback` – logs gradient norms every N steps
- `SVDAttentionCallback` – monitors SVD‑compressed layers

Callbacks are automatically attached by the factory.

### Hyperparameter Optimization

```python
from framework.training import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(train_fn, eval_fn, config_template)
best_params = optimizer.run(n_trials=20)
```

### Metrics

```python
from framework.training import get_task_metrics

compute_metrics_fn = get_task_metrics("mrpc")
# Use with Trainer as `compute_metrics=compute_metrics_fn`
```

## 📊 Run Directory Structure

```
results/
  mrpc_aurora_2026-05-03_10-38-19/
    config.yaml
    metrics.jsonl
    training_history.png
    checkpoints/
      checkpoint-100/
      checkpoint-200/
    logs/
      run.log
    final_model/
      config.json
      pytorch_model.bin
      tokenizer_config.json
      ...
```

## 🔧 Customization

- To add a new trainer, extend the `trainer_factory.py` with your own logic.
- The `ResultsManager` can be used standalone to log metrics from any training loop.

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.

---

*Train, log, and visualize – all in one place.*
