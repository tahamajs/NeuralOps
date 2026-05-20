# `jobscheduler` – Production GPU Job Scheduler

The **jobscheduler** app is a full‑featured, production‑ready GPU job scheduling and management system for the NeuralOps platform.  
It supports **priority queuing with aging**, **backfill**, **fair‑share scheduling across groups**, **graceful preemption with checkpointing**, **dynamic priority adjustment**, and **rich observability** – all powered by a custom, thread‑safe, singleton scheduler that launches training jobs as subprocesses.

---

## Table of Contents

- [`jobscheduler` – Production GPU Job Scheduler](#jobscheduler--production-gpu-job-scheduler)
  - [Table of Contents](#table-of-contents)
  - [Models](#models)
    - [`Job`](#job)
    - [`VersionedConfig`](#versionedconfig)
    - [`CustomMetricEntry`](#custommetricentry)
    - [`JobGroup`](#jobgroup)
  - [Scheduler Architecture](#scheduler-architecture)
    - [Singleton \& Threading](#singleton--threading)
    - [Priority Queue](#priority-queue)
    - [Priority Aging](#priority-aging)
    - [Fair‑Share Scheduling](#fairshare-scheduling)
    - [Backfill](#backfill)
    - [Preemption](#preemption)
    - [Job Execution \& Cleanup](#job-execution--cleanup)
    - [Monitoring](#monitoring)
  - [API Endpoints](#api-endpoints)
  - [Additional Modules](#additional-modules)
    - [Versioning \& Environment Capture (`version_utils.py`)](#versioning--environment-capture-version_utilspy)
    - [Priority Predictor (`predictive.py`)](#priority-predictor-predictivepy)
    - [Dynamic Priority Manager (`dynamic_priority.py`)](#dynamic-priority-manager-dynamic_prioritypy)
    - [Short‑Job Batcher (`batching.py`)](#shortjob-batcher-batchingpy)
    - [Custom Metrics Logger (`metrics.py`)](#custom-metrics-logger-metricspy)
    - [Celery Task (`tasks.py`)](#celery-task-taskspy)
    - [WebSocket Consumer (`consumers.py` \& `routing.py`)](#websocket-consumer-consumerspy--routingpy)
  - [Admin Panel](#admin-panel)
  - [Tests](#tests)
  - [Integration with the Platform](#integration-with-the-platform)
  - [Configuration](#configuration)
  - [Getting Started](#getting-started)

---

## Models

### `Job`
The central entity representing a training, evaluation, or compression job.  
Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `name` | Char | Human‑readable name |
| `model_name`, `task_name`, … | Various | Training configuration (model, task, attention, method, etc.) |
| `gpu_requirements` | Int | Number of GPUs needed |
| `memory_requirements` | Int | Memory in GB |
| `priority` | Int (1‑10) | User‑defined base priority (lower = higher priority) |
| `aging_priority` | Int | Effective priority after aging |
| `group` | Char | Fair‑share group name |
| `preemptible` | Bool | Can this job be preempted? |
| `estimated_duration` | Int | Estimated runtime in seconds |
| `status` | Enum | `queued`, `running`, `completed`, `failed`, `paused`, `preempted` |
| `training_history`, `final_metrics`, `logs` | JSON/Text | Training artifacts |
| `git_commit_hash`, `pip_freeze`, … | Various | Reproducibility metadata |

### `VersionedConfig`
Immutable snapshots of a job’s configuration every time a meaningful change is made (e.g., via the API). Supports diffing between versions.

### `CustomMetricEntry`
Arbitrary key‑value metrics (scalar, image, table, text) logged at specific training steps, allowing rich experiment tracking.

### `JobGroup`
Fair‑share scheduling groups with configurable weights, GPU guarantees, and limits.

---

## Scheduler Architecture

The heart of the app is the `GPUScheduler` – a **thread‑safe singleton** that manages all GPU resources and job execution.  
It is inspired by the best production schedulers: NVIDIA KAI, SLURM, AdaptDL, and DeepPlus.

### Singleton & Threading
- The scheduler is a singleton (`_instance`) that starts automatically when the Django app is loaded (`apps.py`).
- It runs three background daemon threads:
  - **`worker`** – picks jobs from the priority queue and runs them.
  - **`monitor`** – enforces job timeouts.
  - **`backfill`** – periodically scans for short jobs to run in idle GPU gaps.

### Priority Queue
Jobs are held in a `queue.PriorityQueue` sorted by `(effective_priority, submit_time)`.
- Lower numeric effective priority = higher urgency.
- Effective priority can differ from the user‑assigned base priority due to **aging** or the **predictor**.

### Priority Aging
Prevents starvation: a queued job’s effective priority is gradually boosted the longer it waits.

```
effective = base - steps
steps = (wait_time - grace_period) / aging_interval
ceiling: never age beyond HIGH priority (configurable)
```

### Fair‑Share Scheduling
Ensures resources are distributed fairly across **JobGroups**.
- Each group has a **weight** (higher = more resources).
- The scheduler tracks cumulative GPU‑seconds used by each group, decaying with a half‑life (default 24h) to favour recent usage.
- When picking jobs, the effective priority is adjusted by a fairness multiplier:
  ```
  fair_mult = group_weight / (1.0 + recent_usage)
  ```
- This means groups that have used fewer resources recently get a boost.

### Backfill
Inspired by SLURM, backfill allows **short jobs** to jump ahead and use idle GPUs without blocking higher‑priority work.  
A periodic scanner picks queued jobs with `estimated_duration ≤ backfill_max_job_duration` and launches them immediately if free GPUs are available.

### Preemption
When a high‑priority job arrives and no GPUs are free, the scheduler can **preempt** a lower‑priority running job.
- Only jobs with `preemptible = True` are candidates.
- A minimum priority gap (`preempt_min_priority_gap`) must be exceeded to prevent thrashing.
- Before being killed, the target job receives a **signal** (a JSON file written to `/tmp/preempt_flags/`).  
  The training script monitors this file and saves a checkpoint before exiting, enabling later resumption.
- A grace period (`preempt_checkpoint_grace` seconds) is given for the checkpoint to complete; then the process is terminated.

### Job Execution & Cleanup
Jobs are launched via `subprocess.Popen` with a management command.  
- GPU allocation and deallocation are tracked with thread locks.
- After a job finishes (or is preempted), the scheduler frees its GPUs and, if preempted, automatically re‑queues the job.
- Fair‑share usage is recorded for the job’s group.

### Monitoring
- A monitor thread checks for jobs that exceed `job_timeout` and kills them.
- The scheduler exposes a rich `get_status()` dictionary: total GPUs, free GPUs, queue size, running count, fair‑share group usage.

---

## API Endpoints

All endpoints are prefixed with `/api/` and require authentication unless noted.  
Full list:

| Method | URL | Description |
|--------|-----|-------------|
| `GET/POST` | `/api/jobs/` | List all jobs / create a new job |
| `GET/PATCH/DELETE` | `/api/jobs/<uuid:job_id>/` | Get, update, or delete a job |
| `POST` | `/api/jobs/<uuid:job_id>/pause/` | Pause a running job |
| `POST` | `/api/jobs/<uuid:job_id>/resume/` | Resume a paused job |
| `POST` | `/api/jobs/<uuid:job_id>/stop/` | Stop a running job |
| `POST` | `/api/jobs/<uuid:job_id>/retry/` | Retry a failed/stopped job |
| `GET` | `/api/jobs/<uuid:job_id>/history/` | Training history (loss curves) |
| `GET/POST` | `/api/jobs/<uuid:job_id>/notes/` | Get/set researcher notes |
| `GET` | `/api/jobs/<uuid:job_id>/config/history/` | List config versions |
| `GET` | `/api/jobs/<uuid:job_id>/config/compare/<int:va>/<int:vb>/` | Diff two config versions |
| `POST` | `/api/jobs/<uuid:job_id>/metrics/log/` | Log a custom metric |
| `GET` | `/api/jobs/<uuid:job_id>/metrics/` | Retrieve custom metrics |
| `GET` | `/api/jobs/<uuid:job_id>/environment/` | Git/DVC/environment info |
| `PATCH` | `/api/jobs/<uuid:job_id>/scheduling/` | Update priority/duration/group |
| `GET` | `/api/jobs/stats/` | Aggregate job statistics |
| `POST` | `/api/jobs/bulk-delete/` | Delete multiple completed/failed jobs |
| `GET` | `/api/scheduler/status/` | Scheduler status (GPUs, queues) |
| `GET` | `/api/scheduler/metrics/` | Historical completion metrics |
| `GET/POST` | `/api/groups/` | List/create fair‑share groups |
| `GET/DELETE` | `/api/groups/<str:group_name>/` | Get/delete a group |
| `GET` | `/api/scheduler/fairshare/` | Fair‑share usage per group |
| `GET/PATCH` | `/api/scheduler/config/` | Get/update scheduler runtime config |
| `GET` | `/api/system/gpu/` | GPU info (public) |
| `GET` | `/api/system/status/` | System CPU/memory info (public) |
| `GET` | `/api/my-jobs/` | User’s distributed jobs (stub) |

---

## Additional Modules

### Versioning & Environment Capture (`version_utils.py`)
- `capture_git_info(job)` – captures commit hash, branch, dirty status, remote URL.
- `capture_environment(job)` – records Python version and full `pip freeze`.
- `capture_data_version(job)` – captures DVC data hash and Git commit for `.dvc` files.
- `create_config_snapshot(job, ...)` – saves a `VersionedConfig` record when configuration changes.

### Priority Predictor (`predictive.py`)
A **Random Forest** model trained on historical job outcomes (`estimated_duration`, `gpu_requirements`, `task_name`, `actual_priority_outcome`) to suggest optimal priorities.  
- Used by the scheduler to refine priorities at submission time (`0.3*user_prio + 0.7*predicted_prio`).
- Retrained periodically via a Celery task.

### Dynamic Priority Manager (`dynamic_priority.py`)
- Boosts priority of long‑waiting queued jobs.
- Reduces priority of long‑running jobs to free resources for pending tasks.

### Short‑Job Batcher (`batching.py`)
Collects short jobs (duration ≤ threshold) and submits them as a batch with the highest priority among them, improving throughput.

### Custom Metrics Logger (`metrics.py`)
A Python client (`CustomRunLogger`) that sends scalars, images (base64), tables, and text to the `/api/jobs/.../metrics/log/` endpoint during training.

### Celery Task (`tasks.py`)
`retrain_priority_model` – retrains the priority predictor using the latest completed jobs. Schedule via Celery Beat for daily updates.

### WebSocket Consumer (`consumers.py` & `routing.py`)
A `JobLogConsumer` that streams job log file content to a WebSocket, allowing live terminal‑like output in the dashboard.

---

## Admin Panel

All models are registered with Django admin:
- `JobAdmin` – search by name, filter by status/priority, read‑only log fields.
- `VersionedConfigAdmin` – view config history per job.
- `CustomMetricEntryAdmin` – filter by run and type.

---

## Tests

A comprehensive test suite (`tests.py`) covers:
- Job CRUD and validation.
- Scheduling actions (pause, resume, stop, retry).
- Training history and notes.
- Configuration versioning and diffing.
- Custom metric logging and retrieval.
- Environment/Git/DVC info.
- System status and GPU info endpoints.
- Scheduler status and metrics.
- Bulk delete and statistics.
- Error cases (invalid priorities, missing fields, non‑existent jobs, etc.).

Tests use mocked scheduler and versioning to avoid side effects. Run with `python manage.py test jobscheduler`.

---

## Integration with the Platform

The scheduler is designed to work seamlessly with the rest of the NeuralOps stack:
- **Django ORM** – all job state is persisted, allowing the API, admin, and dashboard to remain in sync.
- **Training Runners** – the `run_training_job` management command imports from `framework.experiment.runners` (or directly from the toolkit) to execute the actual ML code.
- **WebSocket** – live updates via Channels groups.
- **Distributed Scheduler** – can be optionally replaced with `DistributedJobScheduler` from `framework.distributed` for multi‑node setups.

---

## Configuration

All scheduler behaviour is controlled via the `SchedulerConfig` dataclass (in `scheduler.py`).  
You can override settings in `django_api/settings.py`:

```python
# scheduler settings
JOB_TIMEOUT_SECONDS = 7200   # kill jobs after 2 hours
```

The runtime configuration can also be inspected and modified via the `/api/scheduler/config/` endpoint.

Key tunables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aging_enabled` | True | Enable priority aging |
| `aging_grace_period` | 300 | Seconds before aging starts |
| `aging_interval` | 60 | Seconds between priority bumps |
| `aging_max_effective` | HIGH | Highest effective priority (never age beyond) |
| `backfill_enabled` | True | Enable backfill scheduling |
| `backfill_interval` | 15 | Seconds between backfill scans |
| `backfill_max_job_duration` | 300 | Max estimated duration for backfill |
| `fair_share_enabled` | True | Enable fair‑share scheduling |
| `fair_share_decay_half_life` | 86400 | Half‑life for usage decay (seconds) |
| `preempt_checkpoint_grace` | 30 | Seconds to allow checkpoint before kill |
| `preempt_min_priority_gap` | 2 | Min priority tiers difference to preempt |
| `job_timeout` | 0 | Job timeout in seconds (0 = disabled) |

---

## Getting Started

1. **Add the app** to `INSTALLED_APPS`:
   ```python
   INSTALLED_APPS += ['jobscheduler']
   ```

2. **Run migrations**:
   ```bash
   python manage.py makemigrations jobscheduler
   python manage.py migrate
   ```

3. **Start the server** – the scheduler starts automatically when the app loads:
   ```bash
   python manage.py runserver
   ```

4. **Create a fair‑share group** (optional):
   ```http
   POST /api/groups/
   {"name": "nlp-lab", "weight": 2.0}
   ```

5. **Submit a training job**:
   ```http
   POST /api/jobs/
   {
     "model_name": "distilbert-base-uncased",
     "task_name": "sst2",
     "attention_type": "galerkin",
     "epochs": 3,
     "gpu_requirements": 1,
     "priority": 3,
     "group": "nlp-lab",
     "preemptible": true
   }
   ```

6. **Monitor** via `/api/scheduler/status/`, `/api/jobs/`, or the admin panel.

