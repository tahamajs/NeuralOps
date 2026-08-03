# `framework/experiment_logging` – Structured Logging & Metrics Tracking

The **experiment_logging** module provides a lightweight, production‑ready logging and metrics tracking system for training experiments.  
It supports **structured (JSON‑line) and human‑readable log formats**, **automatic metrics collection**, and a **HuggingFace Trainer callback**—all configurable via a single dataclass.  
The module integrates seamlessly with the rest of the Efficient Transformer Toolkit (e.g., `ResultsManager`, `MegaOrchestrator`, and the training pipeline).

---

## Quick Start

```python
from framework.experiment_logging import auto_setup, LoggingCallback

# 1. One‑call setup from a run config dictionary
tools = auto_setup(
    run_config={"exp_name": "my_experiment", "log_level": "INFO"},
    run_id="run_001",
    base_dir=Path("./runs")
)

logger = tools["logger"]
tracker = tools["tracker"]

# 2. Attach the callback to your HuggingFace Trainer
trainer.add_callback(LoggingCallback(tracker, logger))

# 3. Train as usual – logs and metrics are captured automatically
trainer.train()
```

After training, the directory `./runs/run_001/logs/` will contain:

- `my_experiment_run_001.log` – rotating log file (structured JSON or human‑readable)
- `metrics.jsonl` – all logged metrics in JSON‑line format

A snapshot of the run configuration is saved to `./runs/run_001/artifacts/config.json`.

---

## Components

### 1. `LoggingConfig` (dataclass)

Holds all logging and tracking settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_level` | `str` | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `log_format` | `str` | `"structured"` | `"structured"` (JSON lines) or `"human"` |
| `log_to_console` | `bool` | `True` | Print logs to stdout |
| `log_to_file` | `bool` | `True` | Write logs to a rotating file |
| `log_dir` | `str` | `"./logs"` | Directory for log files |
| `experiment_name` | `str` | `"experiment"` | Name of the experiment |
| `run_id` | `str` (optional) | `None` | Unique run identifier |
| `use_timestamp` | `bool` | `True` | Append timestamp to file names |
| `metrics_enabled` | `bool` | `True` | Enable metrics tracking |
| `metrics_flush_every` | `int` | `10` | Flush metrics file every N log calls |
| `extra_metadata` | `dict` | `{}` | Key‑value pairs added to every log entry |

### 2. `setup_logging(config: LoggingConfig) -> logging.Logger`

Creates the root logger with the specified configuration.  
It returns the logger for the `"framework"` namespace, which can be used throughout your code.

```python
from framework.experiment_logging import LoggingConfig, setup_logging

cfg = LoggingConfig(log_level="DEBUG", log_format="human")
logger = setup_logging(cfg)
logger.info("Logger ready.")
```

### 3. `get_logger(name: str) -> logging.Logger`

Convenience function to retrieve a logger by name.

### 4. `MetricsTracker`

A simple in‑memory metrics store that can be saved to disk.

```python
tracker = MetricsTracker("./runs/run_001/metrics.jsonl")
tracker.log({"loss": 0.5, "accuracy": 0.8}, step=100)
tracker.flush()  # persists to file
tracker.save()   # explicit save
```

### 5. `LoggingCallback`

A HuggingFace `TrainerCallback` that automatically captures:
- **Training logs** (`on_log`) – logs loss, learning rate, etc. at every logging step.
- **Evaluation metrics** (`on_evaluate`) – logs all metrics returned by `trainer.evaluate()`.
- **Finalisation** (`on_train_end`) – saves all accumulated metrics to disk.

Simply add it to your `Trainer`:

```python
trainer.add_callback(LoggingCallback(tracker, logger))
```

### 6. `auto_setup(run_config, run_id, base_dir) -> dict`

A one‑call convenience that:
- Creates a `LoggingConfig` from the given dictionary.
- Initialises the logger.
- Creates a `MetricsTracker` pointing to the correct log directory.
- Saves a snapshot of the run configuration for reproducibility.

Returns a dictionary with keys `"logger"`, `"tracker"`, and `"config"`.

---

## Structured vs Human‑Readable Format

### Human‑readable (`"human"`)
```
2025-03-01 12:00:00 [INFO] framework: Training started.
2025-03-01 12:00:10 [INFO] framework: Step 100: loss=0.5234 lr=1.85e-05
```

### Structured JSON (`"structured"`)
```json
{"timestamp":"2025-03-01T12:00:00Z","level":"INFO","logger":"framework","message":"Training started."}
{"timestamp":"2025-03-01T12:00:10Z","level":"INFO","logger":"framework","message":"Step 100: loss=0.5234 lr=1.85e-05"}
```

Structured logs are ideal for ingestion into log aggregators like Elasticsearch, Datadog, or for building custom dashboards.

---

## Integration with the Framework

The `experiment_logging` module is used internally by many framework components:

- **`ResultsManager`** – can optionally initialise a `LoggingCallback` to log all metrics.
- **`MegaOrchestrator`** – calls `auto_setup` to set up logging for each run.
- **`run.py`** and **`batch_runner.py`** – use `auto_setup` to create a logger and tracker at the start of every experiment.
- **Training callbacks** – the `LoggingCallback` is automatically added by `HuggingFaceTrainer` when a `ResultsManager` is provided.

You rarely need to call these functions directly when using the high‑level runners.

---

## Dependencies

- `torch` – for `TrainerCallback` (optional, only if using the callback)
- `transformers` – (optional, for the callback)
- Standard library: `json`, `os`, `logging`, `sys`, `datetime`

No external logging libraries are required.

---

## Extending

- **Custom log handler** – add another handler to the root logger after `setup_logging()`, e.g., for sending logs to a remote server.
- **Custom metrics backend** – subclass `MetricsTracker` and override `log()` to send metrics to a database or cloud service.
