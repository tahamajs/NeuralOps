
## `experiment/README.md`

# Experiment App

The `experiment` app is the **core orchestration layer** of the NeuralOps platform.  
It provides:

- **Models** – `Experiment`, `TrainingRun`, `TrainingSnapshot`, `ExperimentTemplate`, `Sweep`, `SweepTrial`
- **REST API** – full CRUD for experiments, training runs, config, artifacts, comparison, analytics, system status, authentication
- **Background Runners** – thread‑based job execution that calls the Efficient Transformer Toolkit
- **WebSocket Consumers** – real‑time progress, metrics, and terminal output
- **Admin Integration** – Django admin registration for all models

All ML work is delegated to the framework (`framework.models`, `framework.training`, `framework.experiment.orchestrator`, etc.). This app only handles the **web layer**.

---

## File Structure

```
experiment/
├── __init__.py
├── apps.py           # AppConfig, ready() hook
├── admin.py          # Django admin registration
├── models.py         # Experiment, TrainingRun, TrainingSnapshot, Sweep, Template
├── serializers.py    # DRF serializers
├── views.py          # All API endpoints
├── urls.py           # URL patterns (prefixed with /api/ in project urls)
├── runners.py        # Background job functions (train, evaluate, compress)
├── jobs.py           # Thread‑based job manager
├── consumers.py      # WebSocket consumers (JobConsumer, TerminalConsumer)
├── routing.py        # WebSocket URL routing
├── signals.py        # Signal handlers (optional)
└── README.md
```

---

## Models

### `Experiment`
- Represents a collection of runs under a common goal.
- Fields: `name`, `description`, `default_config`, `tags`, timestamps.

### `TrainingRun`
- A single training / evaluation / compression job.
- Tracks `status` (queued → running → completed / failed / stopped), `config`, `final_metrics`, `logs`.
- Real‑time fields: `current_step`, `total_steps`, `cpu_usage`, `ram_usage_mb`, `gpu_usage`, `gpu_memory_mb`.
- Property `artifacts_dir` returns the path to saved artifacts (model, plots, etc.).

### `TrainingSnapshot`
- A point‑in‑time metric snapshot (loss, accuracy, LR, GPU memory) at a training step.
- Linked to a `TrainingRun`; `unique_together('run', 'step')` prevents duplicates.

### `ExperimentTemplate`
- A reusable configuration preset that users can save and apply to new experiments.

### `Sweep` / `SweepTrial`
- Hyperparameter sweep support. A `Sweep` has many `SweepTrial` objects, each pointing to a `TrainingRun`.

---

## API Endpoints

All endpoints are prefixed with `/api/` (e.g., `http://localhost:8000/api/runs/`).

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/config/attentions/` | List available attention types |
| `GET` | `/api/config/models/` | List available base models |
| `GET` | `/api/config/tasks/` | List GLUE tasks |
| `GET` | `/api/config/compressors/` | List available compressors |
| `GET` | `/api/config/methods/` | List training methods |
| `GET` | `/api/config/presets/` | List configuration presets |
| `GET/POST` | `/api/experiments/` | List or create experiments |
| `POST` | `/api/experiments/create/` | Create a new experiment |
| `GET` | `/api/experiments/{id}/` | Experiment detail |
| `GET` | `/api/runs/` | List all training runs |
| `GET` | `/api/runs/{id}/status/` | Get a run’s status and details |
| `GET` | `/api/runs/{id}/history/` | Training history (list of metric dicts) |
| `GET` | `/api/runs/{id}/snapshots/` | Per‑step snapshots |
| `PATCH` | `/api/runs/{id}/config/` | Update a run’s configuration |
| `GET` | `/api/runs/{id}/artifacts/` | List saved artifacts |
| `GET` | `/api/runs/{id}/artifacts/{path}` | Download a specific artifact |
| `GET` | `/api/runs/{id}/model/` | Download the trained model as ZIP |
| `POST` | `/api/start_train/` | Start a training job |
| `POST` | `/api/start_evaluate/{id}/` | Start an evaluation job |
| `POST` | `/api/start_compress/{id}/` | Start a compression job |
| `POST` | `/api/stop/{id}/` | Stop a running/queued job |
| `POST` | `/api/stop_all/` | Stop all running jobs |
| `GET` | `/api/runs/{id}/analysis/` | Analysis overview (drift, fairness, etc.) |
| `GET` | `/api/compare_runs/?run_ids=…` | Compare multiple runs |
| `POST` | `/api/compare/` | Compare experiments |
| `GET` | `/api/runs/{a}/diff/{b}/` | Detailed diff between two runs |
| `GET` | `/api/analytics/hp-importance/` | Hyperparameter importance (stub) |
| `GET` | `/api/analytics/metric-correlation/` | Metric correlation (stub) |
| `GET` | `/api/templates/` | List / create templates |
| `GET` | `/api/templates/{id}/` | Template detail |
| `POST` | `/api/templates/{id}/use/` | Apply a template to a run |
| `GET` | `/api/runs/{id}/paper/` | Generate paper (placeholder) |
| `GET` | `/api/runs/{id}/paper_preview/` | PDF preview (stub) |
| `GET` | `/api/runs/{id}/report/` | Experiment report HTML |
| `GET` | `/api/runs/{id}/editor/` | Paper editor page |
| `POST` | `/api/runs/{id}/paper_source/` | Save LaTeX source |
| `GET` | `/api/preview/{id}/` | Preview paper HTML |
| `POST` | `/api/auth/register/` | Register a new user |
| `POST` | `/api/auth/login/` | Login (returns token) |
| `POST` | `/api/auth/logout/` | Logout |
| `GET` | `/api/system/status/` | CPU / RAM usage |
| `GET` | `/api/system/gpu/` | GPU info |

---

## Background Jobs

The `jobs.py` module implements a simple **thread‑based job manager**.  
When a user triggers a training run via `POST /api/start_train/`, the view:

1. Creates a `TrainingRun` record with status `queued`.
2. Calls `start_job(run, run_training)`, which spawns a **daemon thread**.
3. The thread executes `runners.run_training(run, stop_event)`.
4. The runner updates the run’s status, logs, and metrics in real time via `push_update()`, which sends WebSocket messages.

Jobs can be stopped with `POST /api/stop/{id}/`, which sets a `threading.Event` that the runner checks periodically.

---

## Runners

`runners.py` contains the actual execution logic for:

- **`run_training(run, stop_event)`** – launches the `MegaOrchestrator` (or a legacy fallback) that calls the framework’s training pipeline.
- **`run_evaluation(run, stop_event)`** – loads the saved model (or a base model) and runs evaluation using the framework’s `run_evaluation`.
- **`run_compression(run, stop_event)`** – runs compression and saves the compressed model.

Each runner uses the helper `_log` (appends to `run.logs` + WebSocket broadcast) and `_update_status` to keep the database and UI in sync.

---

## WebSocket Consumers

Defined in `consumers.py` and routed via `routing.py`:

- **`JobConsumer`** – joins a group `job_{run_id}` and receives `job_update` messages, broadcasting them to the connected frontend.
- **`TerminalConsumer`** – joins the global `terminal` group and relays `terminal_message` events (used by the PTY terminal app).

The runners call `push_update(run, data)` which sends to the `job_{run_id}` group.

---

## Integration with the Framework

The `experiment` app does **not** contain any ML code itself.  
It imports the framework modules as needed:

- `framework.config.Config` for configuration
- `framework.models.load_model` for loading models with custom attention
- `framework.experiment.orchestrator.MegaOrchestrator` for the full training pipeline
- `framework.training`, `framework.data`, `framework.results_saver` etc.

This separation keeps the web layer clean and the ML layer reusable.

---

## Settings Requirements

In `django_api/settings.py`:

```python
INSTALLED_APPS = [
    'experiment',
    'terminal',
    'rest_framework',
    'rest_framework.authtoken',
    'channels',
    'corsheaders',
    ...
]

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# Channels
ASGI_APPLICATION = 'django_api.asgi.application'
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {'hosts': [('127.0.0.1', 6379)]},
    },
}
```

Redis must be running for WebSocket support.

---

## Running

```bash
cd django_api
python manage.py makemigrations experiment
python manage.py migrate
python manage.py runserver
```

Make sure Redis is running, then start the frontend (`cd frontend && npm run dev`).  
The platform is fully operational at `http://localhost:5173` (frontend) and `http://localhost:8000/api/` (API).

