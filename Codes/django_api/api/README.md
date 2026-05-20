# Backend API & Experiment Runner

Complete **Django REST Framework** backend for the Efficient Transformer Toolkit.  
This application provides:

- User authentication with token‑based (DRF) and cookie fallback
- Real‑time WebSocket updates (job progress, terminal logs)
- Asynchronous **training**, **evaluation**, and **compression** job runners
- Full CRUD for experiments and training runs
- Global activity feed and full‑text search
- System monitoring (CPU, GPU, memory, disk)
- Dynamic configuration endpoints (attention types, compressors, losses, …)
- Automatic model versioning and media serving

All components are loosely coupled and can be dropped into any Django project with minimal configuration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Configuration & Environment](#configuration--environment)
4. [Authentication & User Management](#authentication--user-management)
5. [REST API Endpoints](#rest-api-endpoints)
6. [WebSocket Streams](#websocket-streams)
7. [Experiment Models](#experiment-models)
8. [Job Runners](#job-runners)
9. [Model Versioning](#model-versioning)
10. [Activity & Search](#activity--search)
11. [Deployment](#deployment)
12. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌──────────────┐      HTTP/REST       ┌─────────────────┐
│   Frontend   │ ──────────────────>  │   Django views   │
│  (React/Vue) │ <──────────────────  │  (api/views.py)  │
└──────────────┘      JSON            └────────┬────────┘
                                              │
                                      ┌───────┴────────┐
                                      │   Serializers  │
                                      └───────┬────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │        Django ORM             │
                              │  Experiment, TrainingRun, …   │
                              └───────────────┬───────────────┘
                                              │
           ┌──────────────────────────────────┼──────────────────────────────┐
           │                                  │                              │
   ┌───────┴───────┐                  ┌────────┴────────┐          ┌────────┴────────┐
   │  Job Runner   │                  │  WebSocket      │          │  Activity Log   │
   │ (background   │                  │  (Channels)     │          │  (global feed)  │
   │  threads)     │                  │  real‑time      │          └─────────────────┘
   └───────────────┘                  │  progress       │
                                      └─────────────────┘
```

- **Django + DRF** serves the REST API.
- **Django Channels** powers WebSocket connections for live job updates.
- **Background threads** execute long‑running jobs (training, evaluation, compression) without blocking the API.
- **Framework imports** are lazy – the API can start even if the full ML framework isn’t installed; errors are caught and reported gracefully.

---

## Quick Start

### 1. Requirements

- Python 3.10+
- Django 4.2+
- Django REST Framework
- Django Channels (optional, for WebSockets)
- psutil, torch, numpy (optional – used for system monitoring)
- The Efficient Transformer Toolkit (`framework` package)

### 2. Install Dependencies

```bash
pip install django djangorestframework channels psutil torch numpy
```

### 3. Add the Apps

In your Django project’s `settings.py`, add the required apps:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'channels',                     # if using WebSockets
    'api',
    'experiment',
    # optional: 'model_versioning', 'notes', 'jobscheduler', ...
]
```

### 4. Database Setup

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create a Superuser (optional)

```bash
python manage.py createsuperuser
```

### 6. Attach a Tokenizer to the Model

Most compression and training tasks require a tokenizer. In your Django settings or on model load, attach the tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.tokenizer = tokenizer   # attach as an attribute
```

### 7. Run the Server

```bash
python manage.py runserver
```

The API is now available at `http://localhost:8000/api/`.

---

## Configuration & Environment

All settings are read from Django’s `settings.py`. Key variables:

| Variable             | Default                  | Description                                                   |
|----------------------|--------------------------|---------------------------------------------------------------|
| `MEDIA_ROOT`         | `os.path.join(BASE_DIR, 'media')` | Where job outputs (models, logs) are stored.        |
| `DEBUG`              | `True`                   | Set to `False` in production.                                 |
| `ALLOWED_HOSTS`      | `['*']`                  | Restrict in production.                                       |
| `CACHES`             | `default` backend        | Used for caching model lists and HF search results.           |

For WebSocket support, add the channel layer configuration:

```python
ASGI_APPLICATION = 'your_project.asgi.application'
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',  # or Redis
    },
}
```

---

## Authentication & User Management

The API uses **Token Authentication** from Django REST Framework, with an optional cookie fallback for browser clients. Endpoints:

| Method | URL                       | Description                     |
|--------|---------------------------|---------------------------------|
| POST   | `/api/auth/register/`     | Create new user + return token   |
| POST   | `/api/auth/login/`        | Authenticate user + return token |
| POST   | `/api/auth/logout/`       | Delete token (logout)            |
| GET    | `/api/auth/user/`         | Return current user info         |
| PATCH  | `/api/auth/profile/`      | Update first/last name, email    |
| POST   | `/api/auth/change-password/` | Change password (requires old password) |
| POST   | `/api/auth/refresh-token/`| Issue a new token (old one deleted) |

**Example** – Register a new user:

```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"secure123"}'
```

Response:

```json
{
  "token": "abc123...",
  "username": "alice",
  "user_id": 1
}
```

All subsequent requests must include the token:

```
Authorization: Token abc123...
```

The cookie `authToken` is also set after login/register, so browser clients can authenticate seamlessly.

---

## REST API Endpoints

### Configuration

| Method | URL                            | Description                              |
|--------|--------------------------------|------------------------------------------|
| GET    | `/api/config/attentions/`      | List available attention types           |
| GET    | `/api/config/compressors/`     | List available compression methods       |
| GET    | `/api/config/trainers/`        | List trainer backends                    |
| GET    | `/api/config/methods/`         | List training methods (LoRA, etc.)       |
| GET    | `/api/config/models/`          | List supported base models               |
| GET    | `/api/config/tasks/`           | List supported GLUE tasks                |
| GET    | `/api/config/presets/`         | List presets (fast, accurate, …)         |
| GET    | `/api/config/losses/`          | List loss functions                      |
| GET    | `/api/config/optimizers/`      | List optimizers                          |
| GET    | `/api/config/schedulers/`      | List learning rate schedulers            |
| GET    | `/api/config/callbacks/`       | List training callbacks                  |
| GET    | `/api/config/augmentations/`   | List data augmentation methods           |
| GET    | `/api/config/search-models/`   | Search HuggingFace Hub models            |
| GET    | `/api/config/hpo/`             | HPO options                              |
| GET    | `/api/config/distributed/`     | Distributed training options             |
| GET    | `/api/config/tracking/`        | Experiment tracking options (Wandb, etc.)|
| GET    | `/api/config/deployment/`      | Model deployment options                 |

All configuration endpoints are **read‑only** and **authenticated**. Results are cached for 5 minutes.

### Experiment Management

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/experiments/`              | List all experiments                |
| POST   | `/api/experiments/`              | Create new experiment               |
| GET    | `/api/experiments/{id}/`         | Get experiment details              |
| PATCH  | `/api/experiments/{id}/`         | Update experiment                   |
| DELETE | `/api/experiments/{id}/`         | Delete experiment                   |

### Job (Training Run) Management

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/runs/{id}/`                | Get training run details            |
| POST   | `/api/runs/start-train/`         | Start a training job                |
| POST   | `/api/runs/start-evaluate/`      | Start evaluation job                |
| POST   | `/api/runs/start-compress/`      | Start compression job               |
| POST   | `/api/runs/{id}/stop/`           | Stop a running job                  |
| GET    | `/api/runs/{id}/results/`        | Get training results & artifacts    |
| POST   | `/api/runs/{id}/save/`           | Save run results to permanent storage|

### Mega Launcher

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| POST   | `/api/launch/`                   | Launch full experiment (training + eval + compression + deployment) |

### System Monitoring

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/health/`                   | Health check                        |
| GET    | `/api/version/`                  | API version info                    |
| GET    | `/api/system/status/`            | CPU, memory, disk usage             |
| GET    | `/api/system/gpu/`               | GPU details and utilisation         |

### Load Balancer & GPU Allocation

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/loadbalancer/gpu/status/`  | GPU allocator status                |
| GET    | `/api/loadbalancer/gpu/nodes/`   | Detailed GPU node info              |
| GET    | `/api/loadbalancer/scheduler/status/` | Job scheduler status          |
| GET    | `/api/loadbalancer/scheduler/queue/`  | Pending job queue              |
| GET    | `/api/loadbalancer/scheduler/history/`| Completed job history          |
| GET    | `/api/loadbalancer/cluster/nodes/`| Cluster node information            |
| GET    | `/api/loadbalancer/cluster/gpu-check/`| GPU availability check         |

### Drift, Fairness, Robustness

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/runs/{id}/drift/`          | Get drift report                    |
| POST   | `/api/runs/{id}/drift/check/`    | Run drift detection                 |
| POST   | `/api/runs/{id}/fairness/`       | Run fairness evaluation             |
| POST   | `/api/runs/{id}/robustness/`     | Run adversarial robustness eval     |

### Model Cards & Lineage

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/runs/{id}/model-card/`     | Generate model card markdown         |
| POST   | `/api/runs/{id}/model-card/push/`| Push model card to HuggingFace Hub   |
| GET    | `/api/runs/{id}/lineage/`        | Get data lineage for run             |
| GET    | `/api/models/{id}/lineage/`      | Get model version lineage graph      |

### Prompts (Placeholder)

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/prompts/`                  | List prompt templates               |
| GET    | `/api/prompts/{name}/`           | Get prompt template                 |
| GET    | `/api/prompts/{name}/history/`   | Prompt usage history                |

### Global Search & Activity Feed

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| GET    | `/api/search/`                   | Full‑text search across experiments, notes, models, jobs |
| GET    | `/api/activity/`                 | Latest platform activity            |

### Sub‑Apps (if installed)

| Method | URL                              | Description                         |
|--------|----------------------------------|-------------------------------------|
| *      | `/api/experiments/` (include)    | Experiment app endpoints            |
| *      | `/api/jobs/` (include)           | Job scheduler endpoints             |
| *      | `/api/notes/` (include)          | Researcher notes app                |

---

## WebSocket Streams

Real‑time updates are broadcast to WebSocket clients.

| URL pattern                          | Group          | Description                          |
|--------------------------------------|----------------|--------------------------------------|
| `ws/experiments/{job_id}/`           | `job_{id}`     | Job progress, metrics, status changes|
| `ws/terminal/`                       | `terminal`     | Real‑time log messages               |

**Example connection (JavaScript)**:

```js
const socket = new WebSocket("ws://localhost:8000/ws/experiments/1234/");
socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);  // {type: "progress", progress: 45.2, current_step: 230, total_steps: 512}
};
```

The `JobConsumer` automatically joins the group for the given `job_id`. The `TerminalConsumer` receives all log messages from every job.

---

## Experiment Models

All models live in the `experiment` app.

### Experiment

```python
class Experiment(models.Model):
    id = UUIDField(primary_key=True)
    name = CharField(max_length=255)
    description = TextField(blank=True)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
```

### TrainingRun

```python
class TrainingRun(models.Model):
    id = UUIDField(primary_key=True)
    experiment = ForeignKey(Experiment, related_name='runs', on_delete=CASCADE)
    name = CharField(max_length=255, blank=True)
    job_type = CharField(max_length=20, choices=[('training','training'),('evaluation','evaluation'),('compression','compression')])
    status = CharField(max_length=20, default='queued')
    config = JSONField(default=dict)
    logs = TextField(blank=True)
    final_metrics = JSONField(null=True, blank=True)
    training_history = JSONField(null=True, blank=True)   # list of step dicts
    current_step = IntegerField(default=0)
    total_steps = IntegerField(default=0)
    estimated_remaining_seconds = FloatField(default=0)
    cpu_usage = FloatField(default=0)
    ram_usage_mb = FloatField(default=0)
    started_at = DateTimeField(null=True, blank=True)
    completed_at = DateTimeField(null=True, blank=True)
    created_at = DateTimeField(auto_now_add=True)
```

### TrainingSnapshot

```python
class TrainingSnapshot(models.Model):
    run = ForeignKey(TrainingRun, related_name='snapshots', on_delete=CASCADE)
    step = IntegerField()
    train_loss = FloatField(null=True)
    eval_loss = FloatField(null=True)
    eval_accuracy = FloatField(null=True)
    learning_rate = FloatField(null=True)
    cpu_percent = FloatField(null=True)
    ram_mb = FloatField(null=True)
    timestamp = DateTimeField(auto_now_add=True)
```

### ActivityLog

```python
class ActivityLog(models.Model):
    id = UUIDField(primary_key=True)
    user = ForeignKey(User, on_delete=SET_NULL, null=True)
    action_type = CharField(choices=[...])  # create, update, start, stop, ...
    description = TextField()
    content_type = ForeignKey(ContentType, on_delete=CASCADE, null=True)
    object_id = CharField(max_length=36, null=True)
    metadata = JSONField(default=dict)
    created_at = DateTimeField(auto_now_add=True, db_index=True)
```

Use the helper `log_activity(user, action_type, description, obj)` to create entries.

---

## Job Runners

The `experiment/runners.py` module contains three functions that are executed in **background threads** when a job is started. They integrate with the framework’s actual ML routines.

### Training Runner

```python
run_training(run: TrainingRun, stop_event: threading.Event)
```

- Reads the `config` JSON from the `TrainingRun`.
- Creates an output directory under `MEDIA_ROOT / run.id / model`.
- Calls `framework.experiment.run_training()` with callbacks for progress, logging, and finalisation.
- Progress updates are saved to the database and pushed via WebSocket.
- On completion, automatically creates a `ModelVersion` record if the `model_versioning` app is installed.
- Handles user‑requested stop via the `stop_event`.

### Evaluation Runner

```python
run_evaluation(run: TrainingRun, stop_event)
```

- Loads the trained model from `MEDIA_ROOT / run.id / model` (or falls back to a base model name).
- Calls `framework.experiment.run_evaluation()` to obtain metrics.
- Saves results to `run.final_metrics`.

### Compression Runner

```python
run_compression(run: TrainingRun, stop_event)
```

- Creates the output directory.
- Calls `framework.experiment.run_compression()`.
- Stores compression statistics (e.g., saved parameters, ratio) in `run.final_metrics`.

All runners log messages to `run.logs` and push them to the WebSocket `terminal` group.

---

## Model Versioning

When a training job completes, the runner attempts to import `model_versioning.models.ModelVersion` (if the app is installed). It scans the output directory for `.pth` files, calculates the size, and creates a new version record. This gives you a built‑in model registry with version numbers and metadata.

---

## Activity & Search

### Global Search (`/api/search/`)

Accepts `?q=text&type=all` and searches across:

- `TrainingRun` (name, model_name, task_name)
- `ResearcherNote` (title, content, tags)
- `ModelVersion` (version string)
- `Job` (name, model_name, task_name)

Results are sorted by creation date and limited by `?limit=20`.

### Activity Feed (`/api/activity/`)

Returns the most recent `ActivityLog` entries, including username, action type, description, and link to the entity. This provides a real‑time audit trail for the whole platform.

---

## Deployment

### Production Setup (gunicorn + daphne)

Since the project uses both HTTP and WebSocket, you need an ASGI server. A common pattern:

1. **Static files & media** – served by a reverse proxy (Nginx).
2. **HTTP** – use `gunicorn` with `uvicorn` workers.
3. **WebSocket** – use `daphne` or run the ASGI app directly.

Example `asgi.py`:

```python
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from api.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(websocket_urlpatterns),
})
```

Run with:

```bash
daphne -b 0.0.0.0 -p 8000 your_project.asgi:application
```

### Media Storage

For production, switch to a cloud storage backend (S3, Azure, etc.) using `django-storages`. The runners still write to `MEDIA_ROOT`, but you can mount that to a network file system.

### Security

- Set `SECURE_SSL_REDIRECT = True` and use HTTPS.
- Use Redis for channel layers (instead of in‑memory).
- Run background jobs with a proper task queue (Celery) instead of threads for better scalability.

---

## Troubleshooting

**I get “framework not installed” errors.**  
The API handles missing framework imports gracefully – it returns a `501 Not Implemented` response and logs the error. Install the required packages or update the `PYTHONPATH`.

**WebSocket connections fail.**  
Ensure `channels` is installed and the ASGI routing is correctly configured. Check that your web server supports WebSocket upgrades.

**Training jobs never start.**  
The background threads require the `threading` module. Verify that no global import errors are crashing the thread early – check the Django logs.

**Parameter counts are not updated after compression.**  
Some compressors do not physically remove parameters (they use soft pruning). The `_compressed_params` count reflects the actual tensor sizes, which may remain unchanged. Use the compression stats for effective reduction.

---

## Full API Reference

For a complete interactive documentation, you can install `drf-spectacular` and enable the Swagger UI. However, the endpoint table above covers every route.

