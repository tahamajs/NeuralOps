# 📋 Experiment Logging & Metrics Tracking

**A complete, production‑ready logging system for machine learning experiments.**  
Every training run gets a **timestamped directory**, logs are written in **three formats** (colored console, plain text, structured JSON), and training metrics are **automatically captured** and can be exported to CSV.

---

## 🚀 Quick Start

```python
from experiment_logging import setup_logging, LoggingConfig, MetricsTracker, LoggingCallback

# 1. Configure logging
cfg = LoggingConfig(
    experiment_name="mrpc_aurora",
    log_base_dir="logs",              # top‑level directory
    use_timestamp=True,               # creates unique run folder per run
    json_log=True,                    # structured JSON logs
    console_level="INFO",
    file_level="DEBUG",
)

# 2. Initialize the root logger
logger = setup_logging(cfg)
logger.info("Starting experiment...")

# 3. Metrics tracker (auto‑saves to disk)
tracker = MetricsTracker("metrics.json")   # will be inside the run directory

# 4. Optional: integrate with HuggingFace Trainer
trainer.add_callback(LoggingCallback(tracker, logger))
```

After training, your `logs/` folder will contain:

```
logs/
  mrpc_aurora_2026-05-03_10-38-19/
    run.log          # plain text log
    run.jsonl        # JSON lines log
    metrics.json     # training/evaluation metrics
```

---

## ⚙️ Configuration

All settings are exposed via the `LoggingConfig` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_base_dir` | `"logs"` | Top‑level directory for all runs |
| `experiment_name` | `"experiment"` | Name of the experiment (used in run directory name) |
| `use_timestamp` | `True` | Append current timestamp to the run directory |
| `run_id` | `None` | Explicit run identifier (overrides timestamp if set) |
| `console_level` | `"INFO"` | Minimum logging level for console output |
| `colored_console` | `True` | Colorize console output (requires `colorlog`) |
| `file_level` | `"DEBUG"` | Minimum level for the plain text log file |
| `log_filename` | `"run.log"` | Name of the plain text log file inside the run directory |
| `json_log` | `True` | Enable structured JSON logging (one JSON object per line) |
| `json_filename` | `"run.jsonl"` | Name of the JSON log file inside the run directory |
| `track_metrics` | `True` | Enable metrics tracking (used by `MetricsTracker`) |
| `metrics_filename` | `"metrics.json"` | Default file name for `MetricsTracker` (if no path given) |
| `max_log_size_mb` | `0` | Maximum size (in MB) of the text log before rotation (0 = no rotation) |
| `backup_count` | `5` | Number of backup files when log rotation is active |

You can either pass a `LoggingConfig` object to `setup_logging`, or override fields via environment variables (prefix `LOG_`):

```bash
export LOG_EXPERIMENT_NAME="my_experiment"
export LOG_CONSOLE_LEVEL="DEBUG"
```

---

## 📂 Run Directory

Each call to `setup_logging` creates a **unique run directory**. The naming convention is:

- If `run_id` is set → `{log_base_dir}/{run_id}`
- Else if `use_timestamp=True` → `{log_base_dir}/{experiment_name}_YYYY-MM-DD_HH-MM-SS`
- Else → `{log_base_dir}/{experiment_name}`

The directory is created automatically. We recommend using timestamps (the default) so you never accidentally overwrite previous results.

Inside the run directory you will find:

- **`run.log`** – plain text log with all levels. Useful for `grep` and quick inspection.
- **`run.jsonl`** – structured log; each line is a JSON object with fields: `timestamp`, `level`, `logger`, `message`, `module`, `function`. Perfect for programmatic analysis.
- **`metrics.json`** – optional metrics file (if `MetricsTracker` is used with that path). Each line is a JSON record containing step, loss, accuracy, etc.

---

## 📊 MetricsTracker

`MetricsTracker` records per‑step metrics and flushes them to a JSON‑lines file. It also supports CSV export and simple querying (best loss, etc.).

### Basic Usage

```python
tracker = MetricsTracker("path/to/metrics.json")
tracker.log({"loss": 2.3, "lr": 1e-4}, step=100)
tracker.log({"loss": 2.1, "accuracy": 0.85}, step=200)
tracker.save()                        # forces flush to disk
tracker.to_csv("metrics.csv")         # export to CSV

best = tracker.get_best("loss", mode="min")
print(f"Best loss: {best['loss']} at step {best['step']}")
```

- `log(metrics, step, flush)` – record a dictionary, optionally with a step number. If `flush=True`, the record is immediately written to disk.
- `save()` – flush all in‑memory records to the JSON file.
- `to_csv(path)` – reads all records (from file + memory) and writes a CSV.
- `get_best(metric, mode)` – returns the record with the minimum or maximum value of a given metric.
- `clear()` – empty the in‑memory buffer without affecting the file.

**Important:** The tracker stores data as JSON lines (one JSON per line). This format is easy to parse with `pandas`, `jq`, or any script.

---

## 🔄 Integration with HuggingFace Trainer

The `LoggingCallback` class automatically logs training and evaluation metrics to both the standard logger and a `MetricsTracker`.

```python
from transformers import Trainer
from experiment_logging import LoggingCallback

callback = LoggingCallback(tracker, logger)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[callback],          # ← add the callback
)
trainer.train()
```

- **on_log**: captures `loss`, `learning_rate`, etc., at each logging step.
- **on_evaluate**: captures all evaluation metrics and prints a summary.
- **on_train_end**: forces a final save of all metrics.

---

## 🌈 Colored Console Output

If `colorlog` is installed (`pip install colorlog`), the console output uses colors for different log levels.  
You can disable this by setting `colored_console=False` in the config, or by not installing `colorlog`.

---

## 🧪 Usage from the CLI

The logging system is fully integrated with the Efficient Transformer Toolkit CLI. When you run:

```bash
python cli.py train --preset fast --task mrpc
```

The CLI automatically creates a run directory and starts logging. No extra code needed.

You can also use the logger in standalone scripts:

```python
from experiment_logging import setup_logging, LoggingConfig

cfg = LoggingConfig(experiment_name="my_test")
logger = setup_logging(cfg)
logger.info("Hello world")
```

---

## 🛠️ Advanced Tips

- **Multiple loggers:** Use `get_logger(__name__)` in your submodules to create child loggers. They will inherit the handlers from the root logger.
- **Temporary runs:** Set `use_timestamp=False` and `run_id="debug"` to always write to the same folder during development.
- **Log rotation:** Enable `max_log_size_mb` to rotate the text log file when it exceeds that size (useful for very long runs).
- **Programmatic analysis:** Use the JSON lines log to compute statistics, filter errors, etc.:
  ```python
  import json
  with open("logs/.../run.jsonl") as f:
      for line in f:
          record = json.loads(line)
          if record["level"] == "ERROR":
              print(record["message"])
  ```

---

## 📂 Package Structure

```
experiment_logging/
├── __init__.py          # imports public API
├── config.py            # LoggingConfig dataclass
├── logger.py            # setup_logging, get_logger, formatters
├── tracker.py           # MetricsTracker
├── callbacks.py         # LoggingCallback (for HuggingFace Trainer)
└── README.md            # this file
```

---

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.
