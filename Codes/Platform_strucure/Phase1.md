We're going to take your Phase 1 MVP and break it down into an **exhaustive, implementable blueprint** – every component, every API endpoint, every WebSocket message, every database column, every configuration trick. No code, but so detailed that a team could start building immediately.

---

## 📌 **Phase 1 MVP – Complete Implementation Blueprint**

**Goal:** A working single‑user web dashboard where a researcher can:
1. See a list of all experiments.
2. Launch a new training / evaluation / compression job via a form.
3. Watch live loss curves and logs for any running job.
4. Download artifacts & the trained model when finished.

**Tech Stack (already chosen):**
- Frontend: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Plotly.js, React Query, Zustand, React Router
- Backend: Django 6, DRF, Django Channels, Celery, Redis, SQLite (MVP – later PostgreSQL)
- GPU scheduling: single‑GPU, thread‑based (current `api/jobs.py` mechanism)

---

## 🧱 **Database Schema**

One table – `Experiment` – extended with fields needed for the dashboard.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | primary key |
| `name` | CharField(max_length=200) | user‑given experiment name |
| `job_type` | CharField(choices= ["training","evaluation","compression"]) | what kind of job |
| `status` | CharField(choices= queued→running→completed/failed/stopped) | current state |
| `config` | JSONField | the full experiment config (model, task, attention, hyperparams...) |
| `metrics` | JSONField (default=dict) | latest metrics snapshot (loss, accuracy, …) |
| `logs` | TextField (blank) | accumulated log lines |
| `created_at` | DateTimeField | auto‑now‑add |
| `started_at` | DateTimeField (null) | when job started |
| `completed_at` | DateTimeField (null) | when job finished |
| `artifacts_dir` | CharField | relative path under MEDIA_ROOT |
| `current_step` | IntegerField(default=0) | current training step |
| `total_steps` | IntegerField(default=0) | estimated total steps |
| `estimated_remaining_seconds` | FloatField(null) | ETA |
| `gpu_usage` | FloatField(null) | GPU utilisation % (if available) |

**Indexes:** on `status`, `created_at`, `job_type`.

---

## 🌐 **REST API Endpoints (complete)**

Base URL: `/api`

### 1. Configuration discovery

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/config/models/` | `{ "models": ["distilbert-base-uncased", ...] }` |
| `GET` | `/config/tasks/` | `{ "tasks": ["cola","sst2","mrpc",...] }` |
| `GET` | `/config/attentions/` | `{ "attentions": ["full","aurora","nova",...] }` |
| `GET` | `/config/methods/` | `{ "methods": ["baseline","lora","rdrop"] }` |
| `GET` | `/config/compressors/` | `{ "compressors": ["tt","svd","soka"] }` |
| `GET` | `/config/presets/` | `{ "presets": ["fast","accurate","extreme_compress"] }` |

All return 200, CORS enabled.

### 2. Experiment / Job life‑cycle

| Method | Path | Request Body | Response |
|--------|------|--------------|----------|
| `POST` | `/train/` | Training config (see below) | 202, `Experiment` object |
| `POST` | `/evaluate/` | Evaluation config | 202, `Experiment` object |
| `POST` | `/compress/` | Compression config | 202, `Experiment` object |
| `GET` | `/experiments/` | – | 200, list of all `Experiment` objects |
| `GET` | `/experiments/<id>/` | – | 200, single `Experiment` |
| `PATCH` | `/experiments/<id>/` | fields to update (e.g. `name`) | 200, updated `Experiment` |
| `POST` | `/experiments/<id>/stop/` | – | 200, `{"status":"stopped"}` |
| `GET` | `/experiments/<id>/artifacts/` | – | 200, list of artifact file names |
| `GET` | `/experiments/<id>/artifacts/<path>` | – | 200, file download |
| `GET` | `/experiments/<id>/model/` | – | 200, ZIP file of the final model |

**Training config request body example:**

```json
{
  "name": "mrpc_aurora_test",
  "model_name": "distilbert-base-uncased",
  "task_name": "mrpc",
  "attention_type": "aurora",
  "method": "baseline",
  "compression_rank": 64,
  "epochs": 3,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "max_seq_length": 128,
  "fp16": false,
  "seed": 42
}
```

All missing fields get sensible defaults. The backend stores the whole config and returns an `Experiment` object with `id`.

### 3. System monitoring

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/system/status/` | `{ "cpu": 23.5, "memory": 45.2 }` (percent) |
| `GET` | `/system/gpu/` | `{ "cuda_available": true, "memory_total": 11.76, "memory_used": 2.1, ... }` |

---

## 📡 **WebSocket (Django Channels)**

**Connection URL:** `ws://<host>/ws/experiments/<id>/`

After connecting, the server immediately sends the current status and logs. Subsequently, the server pushes messages as training progresses.

**Server → Client message types:**

| `type` | Payload | When |
|--------|---------|------|
| `"status"` | `{ "status": "running" }` | job status change |
| `"log"` | `{ "data": "Epoch 1/3 ..." }` | new log line |
| `"metrics"` | `{ "loss": 0.234, "accuracy": 0.89, ... }` | trainer logs |
| `"progress"` | `{ "current_step": 100, "total_steps": 1000, "eta_seconds": 360 }` | every 10 steps |
| `"gpu"` | `{ "memory_used_gb": 2.3, "utilization": 85 }` | optional, periodic |

The frontend maintains a single WebSocket per viewed experiment.

---

## 🎨 **Frontend – Detailed Component Tree**

### **Routes**

```
/                           → redirect to /dashboard
/dashboard                  → DashboardPage
/experiments                → ExperimentListPage
/experiments/:id            → ExperimentDetailPage
/launcher                   → LauncherPage (training/eval/compress)
```

### **DashboardPage**

- **State:** `experiments` (via `useQuery`)
- **Components:**
  - `MetricSummaryCards` – count of queued, running, completed, failed
  - `RecentExperimentsTable` – last 10 experiments, clickable
  - `ActiveJobsWidget` – list of running/queued jobs with mini progress bars
  - `SystemResourcesWidget` – polls `/api/system/` every 5s, shows CPU/RAM, GPU info

### **ExperimentListPage**

- **State:** `experiments` (React Query, auto‑refresh)
- **Components:**
  - `ExperimentFilters` – filter by status, job type, search by name
  - `ExperimentTable` – columns: Name, Type, Status, Created, Actions (View, Stop, Download)
  - `PaginationControls`

### **ExperimentDetailPage**

- **State:** `experiment` (React Query), live metrics via WebSocket
- **Components:**
  - `StatusBadge` – colour‑coded status
  - `ProgressBar` – with ETA text
  - `LiveLossPlot` – Plotly line chart, updates on each `"metrics"` message
  - `LiveLogViewer` – scrolling log pane, auto‑scroll
  - `ConfigSummary` – read‑only YAML / JSON view of the config
  - `ArtifactList` – fetched from `/artifacts/`, download links
  - `DownloadModelButton` – triggers `/model/` ZIP download

### **LauncherPage**

- **State:** local form state, `useMutation` to submit
- **Sub‑pages:** training, evaluation, compression (selected via tabs)
- **Components (training tab):**
  - `ModelPicker` – select dropdown, fetches `/config/models/`
  - `TaskPicker` – select dropdown, fetches `/config/tasks/`
  - `AttentionPicker` – select dropdown, fetches `/config/attentions/`
  - `MethodPicker` – select dropdown (baseline, lora, rdrop…)
  - `HyperparameterFields` – number inputs for epochs, batch size, learning rate, rank, max length
  - `PresetSelector` – loads a predefined config into all fields
  - `ConfigPreviewPanel` – shows the JSON that will be sent
  - `LaunchButton` – validates, calls mutation, on success navigates to the experiment detail page
- **Evaluation tab:** model path (or pick from registry), task, batch size.
- **Compression tab:** model name, compression method, target rank.

---

## ⚙️ **Backend Implementation Details (Django)**

### **Project structure additions**

```
api/
├── models.py           # Experiment model (extended)
├── serializers.py      # ExperimentSerializer
├── views.py            # all DRF views
├── urls.py             # URL routing
├── consumers.py        # WebSocket consumer
├── routing.py          # WebSocket routing
├── jobs.py             # thread‑based job manager (current)
├── runners.py          # training/eval/compression logic
└── tasks.py            # (future) Celery tasks – we'll keep threading for MVP
```

### **Background job execution flow**

1. View creates `Experiment` with `status="queued"`.
2. View calls `start_job(experiment, runner_func)` (from `jobs.py`).
3. `start_job` spawns a daemon thread that sets `status="running"`, calls the runner, captures exceptions, updates status and logs.
4. Runner (`runners.py`) uses the framework's `CompressionTrainer` with a custom callback that:
   - Saves logs to the `logs` field (batched, every line)
   - Pushes metrics & progress via WebSocket using `push_update`
   - Updates `current_step`, `total_steps`, `eta_seconds`, `cpu_usage`, `ram_usage_mb` on the model
5. On completion, saves the model to `MEDIA_ROOT/<id>/model/` and marks status `completed`.

### **WebSocket push helper**

```python
def push_update(experiment, message):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"experiment_{experiment.id}",
        {"type": "send_message", "message": message}
    )
```

Consumer joins group `experiment_<id>` and forwards messages to the client.

### **Artifact serving**

Artifacts (plots, logs, etc.) are saved in `MEDIA_ROOT/<id>/`. The API lists files and serves them via `FileResponse`. The model is zipped on‑the‑fly when requested.

---

## 📊 **React State Management Plan**

- **Server state** (experiments list, single experiment, config options) → **React Query** with `staleTime: 10s` for lists, `0` for details.
- **Live experiment data** (metrics, logs, progress) → **Zustand store** updated by WebSocket listener. The store holds:
  ```ts
  {
    status: string,
    logs: string[],
    metrics: {loss: number[], accuracy: number[]},
    progress: {current: number, total: number, eta: number},
    gpuInfo: {memoryUsed: number, total: number}
  }
  ```
- **Form state** (launcher) → `useReducer` + React Hook Form with Zod schema.

---

## 🎯 **Step‑by‑Step Implementation Order (MVP)**

### Week 1

1. **Backend:** Extend `Experiment` model, add serializers, views for `POST /train/`, `POST /evaluate/`, `POST /compress/`, `GET /experiments/`, `GET /experiments/<id>/`, `PATCH`, stop, artifacts, model download.
2. **Backend:** Implement `system/` endpoints for GPU/CPU monitoring.
3. **Backend:** Update `runners.py` to use extended model and push live data.
4. **Backend:** Verify training jobs run correctly with live logs & metrics.
5. **Frontend:** Scaffold React project with routing, shadcn/ui, React Query.
6. **Frontend:** Build `ExperimentListPage` with table, filters, pagination.
7. **Frontend:** Build `LauncherPage` for training (model, task, attention, HP fields, presets).
8. **Integration:** Connect launch form to API, navigate to detail page on success.

### Week 2

9. **Frontend:** Build `ExperimentDetailPage` with status, progress bar, Plotly loss chart, log viewer.
10. **Frontend:** Implement WebSocket connection and live updates (metrics, logs).
11. **Frontend:** Add artifact list and model download.
12. **Frontend:** Build simple `DashboardPage` with summary cards and recent experiments table.
13. **Testing:** End‑to‑end flow: launch training → watch live → artifacts → model download.
14. **Polish:** Empty states, loading skeletons, error boundaries, responsive design.

---

## 🔮 **What Phase 1 Delivers**

- A working web dashboard to manage compression experiments.
- Ability to launch training, evaluation, and compression jobs from the UI.
- Real‑time monitoring of loss, accuracy, progress, and hardware usage.
- Downloadable artifacts and trained models.
- All state persisted in a database, logs streamed live.

Phase 1 is a complete, self‑contained MVP that can be used productively by a single researcher. The architecture is designed to scale seamlessly into Phase 2 (multi‑GPU, sweeps, model registry) without rewriting any core component.