# 🏋️ Training Module

The training module provides a **unified, extensible interface** for training Transformer models with the Efficient Transformer Toolkit.  
It supports standard Hugging Face `Trainer`, Unsloth‑accelerated training, Reinforcement Learning (via TRL), and custom training loops – all with automatic result logging and versioning.

## ✨ Features

- **Registry‑based trainer selection** – add new trainers without modifying the factory.
- **Automatic result management** – timestamped directories, config saves, metrics (JSONL), logs, final model, training plots.
- **Callbacks** – memory usage, elapsed time, gradient norms, SVD attention stats.
- **Hyperparameter optimization** – Optuna integration with pruning and visualisations.
- **GLUE metrics** – ready‑to‑use accuracy, F1, Matthews corr., Pearson/Spearman.
- **Lazy imports** – optional dependencies (`unsloth`, `trl`, `optuna`) are imported only when needed.

## 📁 Structure

```
framework/training/
├── __init__.py              # public API
├── base_trainer.py          # abstract base class
├── registry.py              # trainer registry
├── trainer_factory.py       # factory using registry
├── hf_trainer.py            # Hugging Face Trainer
├── unsloth_trainer.py       # Unsloth‑accelerated
├── rl_trainer.py            # TRL PPO
├── custom_loop_trainer.py   # custom PyTorch loop
├── trainer.py               # SVDAwareTrainer (extends HF)
├── callbacks.py             # Memory, Time, GradientNorm, SVDAttention
├── metrics.py               # per‑GLUE metrics
├── utils.py                 # custom training/evaluation/benchmark
├── results_manager.py       # directory & logging management
└── hyperopt.py              # Optuna hyperparameter optimisation
```

## 🚀 Quick Start

```python
from framework.training import get_trainer

config = {
    "experiment_name": "mrpc_aurora",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
}

trainer, results_mgr = get_trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config=config,
    trainer_type="hf",           # "hf", "unsloth", "trl", "custom_loop"
    device=device,
    num_labels=2,
    compute_metrics=my_metrics_fn,
)

trainer.train()
```

All outputs go to `results/mrpc_aurora_<timestamp>/`:
- `config.yaml`
- `metrics.jsonl`
- `training_history.png`
- `checkpoints/`
- `logs/`
- `final_model/`

## 🧩 Trainer Types

| `trainer_type` | Backend | Dependencies | Use Case |
|----------------|---------|--------------|----------|
| `"hf"` (default) | Hugging Face `Trainer` + `SVDAwareTrainer` | `transformers` | General purpose |
| `"unsloth"` | Unsloth `FastLanguageModel` + `Trainer` | `unsloth` | Faster training on consumer GPUs |
| `"trl"` | TRL `PPOTrainer` | `trl` | Reinforcement Learning from Human Feedback |
| `"custom_loop"` | Pure PyTorch loop | none | Full control, debugging |

## 📈 Hyperparameter Optimisation

```python
from framework.training import HyperparameterOptimizer

config_template = {...}
optimizer = HyperparameterOptimizer(train_fn, eval_fn, config_template)
best_params = optimizer.run(n_trials=20)
```

## 🔧 Custom Callbacks

All provided callbacks are automatically attached by `ResultsManager.wrap_trainer()`.  
You can add your own:

```python
from transformers.trainer_callback import TrainerCallback

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # custom logic
        pass

trainer.add_callback(MyCallback())
```

## 📊 Metrics

Use `get_task_metrics(task_name)` to obtain the correct metric function for a GLUE task:

```python
from framework.training import get_task_metrics

compute_metrics = get_task_metrics("mrpc")  # returns {'accuracy', 'f1'}
```

## 🧪 Testing

Run the training module with a small dataset to verify everything works:

```bash
python -c "from framework.training import get_trainer; print('OK')"
```

## 🤝 Extending

1. Create a new class inheriting from `BaseTrainer`.
2. Decorate it with `@TrainerRegistry.register("my_trainer")`.
3. Implement `train()` and `evaluate()`.
4. The factory `get_trainer(trainer_type="my_trainer")` will now work.

## 📄 License

Part of the **Efficient Transformer Toolkit** – MIT License.
