We now push **Phase 2** into the same level of granularity as Phase 1 – every component, every API endpoint, every database migration, every UX state, every WebSocket enhancement, every Optuna integration detail, every model‑registry lifecycle, and every GPU‑cluster visualization widget.

---

# 🔥 Phase 2 – Advanced Experiment Platform

**Timeline:** 3 weeks  
**Prerequisites:** Phase 1 fully functional (experiment list, launch form, live plot, log viewer, artifact download)

**New capabilities delivered:**

| Domain | What’s New |
|--------|------------|
| **Experiment Tracking (W&B‑like)** | Config diffs, metric comparison charts, lineage graph, reproducibility snapshots, run grouping/tagging |
| **Hyperparameter Sweeps** | Optuna‑powered sweep creation, trial management, parallel‑coordinate plot, importance chart, early‑stopping (Hyperband) |
| **Model Registry** | Model versioning, metadata tagging, deployment status, model card, download/deploy actions |
| **Real‑time WebSocket streaming** | Throttled streaming, metrics compression, GZIP, reconnect with replay, per‑channel authentication |
| **GPU Cluster Monitoring** | Node‑level GPU utilization heatmap, memory pressure gauge, job queue depth, Prometheus‑compatible metrics endpoint |

---

## 1. 📊 Experiment Tracking (W&B‑Style)

### 1.1 New Database Models

```
ExperimentRun (extends the MVP "Experiment")
├── id (UUID, PK)
├── name (CharField)
├── project (ForeignKey → Project)        ← NEW
├── tags (ManyToMany → Tag)               ← NEW
├── job_type (choices)
├── status (CharField)
├── config (JSONField)
├── metrics_snapshot (JSONField)          ← snapshot every N steps
├── logs (TextField)
├── started_at / completed_at
├── parent_run (self‑FK, nullable)        ← for sweep lineage
├── artifacts_dir
├── current_step / total_steps
└── is_favorite (BooleanField)            ← user can star runs
```

```
Project
├── id
├── name (unique)
├── description
└── created_at
```

```
Tag
├── id
└── name (unique)
```

```
Sweep (NEW)
├── id
├── project (FK)
├── name
├── config (JSONField)       – search_space, algorithm, max_trials, etc.
├── status (active/paused/completed)
├── created_at
```

```
SweepTrial (NEW)
├── id
├── sweep (FK)
├── run (FK → ExperimentRun)   ← each trial is an ExperimentRun
├── trial_number
├── status
├── objective_value (Float)
└── params (JSONField)         ← actual hyperparameters sampled
```

### 1.2 New REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/projects/` | List projects |
| `POST` | `/projects/` | Create project |
| `GET` | `/runs/` | List runs (filter by project, tags, status, job_type, date) |
| `GET` | `/runs/<id>/` | Single run detail (extended with metrics history) |
| `GET` | `/runs/compare/?run_ids=a,b,c` | Return config diff + metrics overlay data |
| `PATCH` | `/runs/<id>/` | Update tags, star status |
| `DELETE` | `/runs/<id>/` | Soft‑delete |
| `GET` | `/runs/<id>/metrics?keys=loss,accuracy` | Return metric time‑series (from metrics_snapshot) |

### 1.3 Config Diff Engine

When the frontend requests `/runs/compare/?run_ids=a,b,c`:
- Backend loads the three `config` JSON blobs.
- Flattens them (dotted keys).
- Returns a dict:
  ```json
  {
    "keys": ["learning_rate", "batch_size", "attention_type", ...],
    "runs": {
      "a": {"learning_rate": 2e-5, "batch_size": 16, ...},
      "b": {"learning_rate": 5e-5, "batch_size": 16, ...},
      "c": {"learning_rate": 2e-5, "batch_size": 32, ...}
    }
  }
  ```
- Frontend renders a table. Differences are highlighted in yellow.

### 1.4 Metric Comparison Chart

The endpoint `/runs/compare/?run_ids=a,b,c` also returns metrics time‑series:
```json
{
  "metrics": {
    "loss": {
      "a": [[step, val], [step, val], ...],
      "b": [[step, val], ...],
      "c": [[step, val], ...]
    },
    "accuracy": { ... }
  }
}
```
Frontend overlays all three loss curves on one Plotly chart, with legend per run.

### 1.5 Reproducibility Snapshot

At the end of each run, the backend serializes:
- Full config
- Git commit hash (extracted via `subprocess` or passed in config)
- Python environment (`pip freeze` → stored as artifact)
- Random seed

This is stored as a JSON blob in `artifacts_dir/reproducibility.json`. The frontend shows a "Reproduce" button that downloads this file and optionally a script to rerun.

### 1.6 Frontend Components (new)

| Component | Description |
|-----------|-------------|
| `ProjectSelector` | Dropdown to filter by project |
| `RunTable` | Sortable columns (name, status, metrics, created, tags). Multi‑select for compare. |
| `CompareDrawer` | Slides in from right – shows config diff table + metric overlay chart |
| `TagInput` | Autocomplete input for adding/removing tags |
| `MetricExplorer` | Select metric keys, plot multiple runs, zoom, export PNG/SVG |
| `ReproducibilityPanel` | Shows git hash, seed, environment; download button |

---

## 2. 🔬 Hyperparameter Sweeps (Optuna)

### 2.1 Architecture

```
Frontend                    Backend                         Celery Worker
────────                    ───────                         ─────────────
SweepCreator → POST /sweeps/ → create Sweep + enqueue study
                          ↓
                    Celery Beat picks up pending trials
                          ↓
            For each trial:
              1. Optuna suggests params
              2. Creates ExperimentRun (trial)
              3. Enqueues training Celery task
              4. On completion → reports objective value back to Optuna
              5. Checks pruning (Hyperband)
```

### 2.2 New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sweeps/` | Create a new sweep. Body: `{ "project_id": ..., "name": ..., "config": {...} }` |
| `GET` | `/sweeps/` | List sweeps (filter by project) |
| `GET` | `/sweeps/<id>/` | Sweep detail (status, best trial, trials list, parallel‑coord data) |
| `POST` | `/sweeps/<id>/pause/` | Pause a sweep |
| `POST` | `/sweeps/<id>/resume/` | Resume a paused sweep |
| `GET` | `/sweeps/<id>/trials/` | List trials with status and objective value |
| `GET` | `/sweeps/<id>/importance/` | Return parameter importance (Optuna built‑in) |

### 2.3 Sweep Configuration Schema

```json
{
  "search_space": {
    "learning_rate": {"type": "float", "low": 1e-6, "high": 5e-5, "log": true},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
    "rank": {"type": "int", "low": 32, "high": 256, "step": 32},
    "attention_type": {"type": "categorical", "choices": ["aurora", "nova", "full"]}
  },
  "algorithm": "tpe",           // tpe | random | grid
  "direction": "maximize",
  "metric": "eval_accuracy",
  "max_trials": 50,
  "pruning": "hyperband",       // none | median | hyperband
  "parallel_workers": 4,
  "early_stopping_min_trials": 10
}
```

### 2.4 Backend Sweep Engine

A Celery task `run_sweep` manages the study:
- Creates an Optuna `Study` backed by a persistent storage (SQLite for MVP, PostgreSQL later).
- For each trial:
  1. Calls `study.ask()` to get suggested params.
  2. Copies the base config and overrides with suggested params.
  3. Creates an `ExperimentRun` linked to the `SweepTrial`.
  4. Submits a training Celery task.
  5. On training completion, gets the objective value, calls `study.tell(trial, value)`.
  6. If pruning enabled, checks `trial.should_prune()` after intermediate reports.

### 2.5 Frontend Components (Sweep)

| Component | Description |
|-----------|-------------|
| `SweepCreator` | 3‑step wizard: (1) select base config, (2) define search space, (3) configure algorithm & budget |
| `SearchSpaceEditor` | Dynamic form – add parameters, choose type, set ranges. Preview JSON. |
| `SweepDashboard` | Parallel‑coordinates plot (Plotly), trial status table, best‑trial highlight, pause/resume button |
| `TrialTable` | Columns: trial #, params, objective value, status, runtime. Clickable – navigates to the ExperimentRun detail. |
| `ParameterImportanceChart` | Bar chart from Optuna's `get_param_importances()` |
| `OptimizationHistoryChart` | Line chart of best objective value vs. trial number |

### 2.6 Optuna Integration with Pruning (Hyperband)

In `runners.py`, add a pruning callback that reports intermediate metrics to Optuna:
```python
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, study):
        self.trial = trial
        self.study = study
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.trial.report(metrics["eval_accuracy"], step=state.global_step)
            if self.trial.should_prune():
                control.should_training_stop = True
```
This enables Hyperband to kill underperforming trials early, saving GPU hours.

---

## 3. 🏛️ Model Registry

### 3.1 Database Models

```
ModelVersion
├── id (UUID)
├── name (CharField)                        – e.g. "mrpc-aurora-v1"
├── version_number (IntegerField)           – auto‑increment per model
├── run (FK → ExperimentRun)                – source training run
├── framework (CharField)                   – "transformers"
├── task_type (CharField)
├── metrics (JSONField)                     – final evaluation metrics
├── model_file_path (CharField)             – S3 / local path to model.bin
├── tokenizer_file_path (CharField)
├── config_snapshot (JSONField)             – full model config at save time
├── tags (ManyToMany → Tag)
├── deployment_status (CharField)           – none/staging/production/archived
├── deployed_at (DateTimeField, null)
├── created_at
└── created_by (FK → User, null for MVP)
```

### 3.2 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/models/` | Register a new model version (from a completed run) |
| `GET` | `/models/` | List all model versions (filter by name, tags, status) |
| `GET` | `/models/<id>/` | Detail view including metrics, config, deployment status |
| `PATCH` | `/models/<id>/` | Update deployment status or tags |
| `GET` | `/models/<id>/download/` | Download model artifacts (ZIP) |
| `POST` | `/models/<id>/deploy/` | Mark as production, optionally triggers inference endpoint |

### 3.3 Auto‑Registration Hook

When a training run completes successfully (`status="completed"`), the backend automatically creates a `ModelVersion`:
- Extracts final metrics from the run's `metrics_snapshot`.
- Copies the model directory to a permanent location under `MEDIA_ROOT/models/<name>/<version>/`.
- Sets `deployment_status="none"`.

### 3.4 Frontend Components (Registry)

| Component | Description |
|-----------|-------------|
| `ModelTable` | Columns: name, version, task, metrics, deployment status, created, actions |
| `ModelDetail` | Full metadata, metric history chart, download button, deploy button |
| `DeployModal` | Select deployment target (staging/production), confirm |
| `ModelComparison` | Select two versions, show metric diff table side‑by‑side |

---

## 4. 📡 Enhanced Real‑Time WebSocket Streaming

### 4.1 WebSocket Protocol Enhancements

**Message types added in Phase 2:**

| Type | Payload | Purpose |
|------|---------|---------|
| `"heartbeat"` | `{"timestamp": 1714764820}` | Every 15s to keep connection alive |
| `"metrics_batch"` | `{"loss": [...], "accuracy": [...], "step": [...]}` | Compressed batch of 10 steps sent at once |
| `"replay"` | Full metrics history on connect | Frontend replays missed data after reconnect |
| `"error"` | `{"message": "GPU OOM", "timestamp": ...}` | Critical errors pushed immediately |

### 4.2 Performance Optimizations

| Feature | Implementation |
|---------|----------------|
| **Throttling** | Push metrics at most every 2 seconds. Batch intermediate values. |
| **GZIP compression** | Django Channels supports GZIP for WebSocket frames. |
| **Metrics delta** | Send only the last N (e.g., 50) data points on connect; send only new points thereafter. |
| **Channel layer** | Switch from InMemory to Redis for production multi‑process. |

### 4.3 Frontend Reconnect Strategy

- **Exponential backoff:** on disconnect, retry at 1s, 2s, 4s, … max 30s.
- **On reconnect:** request replay via a special WebSocket message `{"type": "replay_request"}`. Server sends full metrics history since last disconnect.
- **Zustand store** merges replayed data into existing traces.

---

## 5. 🖥️ GPU Cluster Monitoring Dashboard

### 5.1 Data Collection

| Data Source | What it provides |
|-------------|------------------|
| **NVIDIA DCGM** | GPU utilization, memory, temperature, power, SM occupancy, ECC errors |
| **Prometheus Node Exporter** | CPU, RAM, disk, network per node |
| **Django custom metrics** | Celery queue lengths, job counts, experiment statuses |
| **Celery events** | Worker heartbeats, task start/finish timestamps |

### 5.2 New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/cluster/nodes/` | List all GPU nodes with real‑time metrics |
| `GET` | `/cluster/nodes/<name>/` | Single node detail (GPUs, processes, memory breakdown) |
| `GET` | `/cluster/queue/` | Celery queue depths (training, evaluation, compression) |
| `GET` | `/cluster/jobs/` | Active jobs per worker |
| `GET` | `/metrics/` (Prometheus endpoint) | Prometheus‑compatible text format for scraping |

### 5.3 Frontend Components (Cluster)

| Component | Description |
|-----------|-------------|
| `GPUNodeCard` | Single node: hostname, GPU count, avg utilization bar, memory used/total, status dot (green/yellow/red) |
| `GPUUtilHeatmap` | Grid of all GPUs across all nodes, color‑coded by utilization |
| `NodeDetailPanel` | Per‑GPU metrics (SM, memory, temp, power), process list with user/PID/memory |
| `QueueDepthChart` | Time‑series of pending/running jobs per queue |
| `JobTimeline` | Gantt chart of currently running jobs, with stage breakdown |

### 5.4 Integration with Kueue / Run:ai (Future)

For Kubernetes‑based GPU scheduling:
- Deploy **Kueue** to manage GPU quotas and priorities.
- Deploy **Run:ai** (or **Volcano**) for fair‑share scheduling and GPU fractioning.
- The frontend submits jobs with `priority_class` and `gpu_limit` fields.
- The cluster dashboard pulls metrics from Kueue API to show queue status and wait times.

---

## 6. 🏗️ Phase 2 Implementation Schedule

### Week 1 – Experiment Tracking & Model Registry

1. **Backend:** Add `Project`, `Tag`, `ModelVersion` models. Migrations.
2. **Backend:** Implement config diff engine and metric comparison endpoint.
3. **Backend:** Auto‑registration hook for ModelVersion on run completion.
4. **Frontend:** Build `ProjectSelector`, `RunTable` with multi‑select, `CompareDrawer` with config diff table.
5. **Frontend:** Build `MetricExplorer` for multi‑run overlay charts.
6. **Frontend:** Build `ModelTable`, `ModelDetail`, `DeployModal`.

### Week 2 – Hyperparameter Sweeps

7. **Backend:** Add `Sweep`, `SweepTrial` models. Sweep engine with Optuna integration.
8. **Backend:** Implement sweep creation, pause/resume, trial management endpoints.
9. **Backend:** Add pruning callback in runners.py.
10. **Frontend:** Build `SweepCreator` wizard (3 steps).
11. **Frontend:** Build `SweepDashboard` with parallel‑coordinates, trial table, importance chart.

### Week 3 – Cluster Monitoring & Polish

12. **Backend:** Add cluster nodes, queue, Prometheus endpoints.
13. **Backend:** Enhance WebSocket with replay, batching, heartbeat.
14. **Frontend:** Build GPU cluster dashboard (node grid, heatmap, queue chart).
15. **Frontend:** Implement WebSocket reconnect with replay.
16. **Testing:** End‑to‑end sweep (create sweep → trials run → compare results → register best model).
17. **Polish:** Responsive design audit, empty states, error boundaries, loading skeletons.

---

## 7. 🔮 What Phase 2 Delivers

- A complete W&B‑like experiment tracking platform with project organization, tagging, config diffs, and metric comparison.
- Automated hyperparameter optimization via Optuna sweeps with Bayesian search, Hyperband pruning, and interactive visualization.
- A versioned model registry with metadata, deployment status, and one‑click model serving.
- Robust real‑time streaming with replay, batching, and reconnect.
- GPU cluster monitoring dashboard showing node health, utilization heatmaps, and queue depths.

At the end of Phase 2, you have a **production‑grade ML experiment platform** ready for a small team (3‑10 researchers). The architecture supports seamless transition to Phase 3 (multi‑node Kubernetes, Ray Train, and auto‑scaling).