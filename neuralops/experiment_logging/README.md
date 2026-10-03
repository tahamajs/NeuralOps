# Experiment Logging Subsystem (`Codes/framework/experiment_logging`)

The `experiment_logging` subsystem provides structured JSON logging, distributed log aggregation, execution trace capture, and real-time streaming telemetry for high-throughput experiments.

---

## 🏗 Architecture Overview

- **Structured Logger (`logger.py`)**: Asynchronous, thread-safe logger writing structured JSON-Lines events with timestamping, process IDs, and rank identifiers.
- **Distributed Aggregator (`logger.py`, `integration.py`)**: Gathers log streams across multi-GPU DDP ranks, filtering redundant outputs and routing primary rank logs.
- **Training Callbacks (`callbacks.py`)**: Live event listeners logging epoch summaries, throughput telemetry (samples/sec, tokens/sec), and memory profiles.
- **Config & Context (`config.py`)**: `LoggingConfig` configuring log levels, output destinations, JSON formatting, and rotation policies.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `logger.py` | Core structured logger with JSON formatting and distributed safety. |
| `callbacks.py` | Training loop logging callbacks for metric recording and milestone notifications. |
| `config.py` | `LoggingConfig` dataclass managing log files, formatting, and verbosity. |
| `integration.py` | Connects structured logging into trainers, distributed runners, and evaluation pipelines. |

---

## 🚀 Quick Start & Usage

### 1. Setting up Structured JSON Logging
```python
from framework.experiment_logging.logger import get_experiment_logger
from framework.experiment_logging.config import LoggingConfig

config = LoggingConfig(log_dir="./logs", log_to_json=True, level="INFO")
logger = get_experiment_logger("training_run", config=config)

logger.info("Model training initiated", extra={"model": "distilbert", "rank": 32})
```

### 2. Logging Training Milestones
```python
logger.info("Epoch completed", extra={"epoch": 1, "loss": 0.42, "accuracy": 0.85})
```
