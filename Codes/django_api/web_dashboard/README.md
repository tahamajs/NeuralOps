# `web_dashboard` – Complete Customizable Dashboard App

The **web_dashboard** app is the **central user interface** of the NeuralOps platform. It provides a **drag‑and‑drop dashboard** with live data widgets, **JWT authentication**, **real‑time WebSocket updates**, and deep integration with every other app in the ecosystem – `experiment`, `jobscheduler`, `sweeps`, `terminal`, `model_versioning`, and `notes`.

---

## Table of Contents

- [`web_dashboard` – Complete Customizable Dashboard App](#web_dashboard--complete-customizable-dashboard-app)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Features at a Glance](#features-at-a-glance)
  - [Models](#models)
    - [`DashboardWidget`](#dashboardwidget)
  - [Authentication](#authentication)
    - [Endpoints](#endpoints)
  - [Widget System](#widget-system)
    - [Widget CRUD](#widget-crud)
    - [Widget Configuration](#widget-configuration)
    - [Default Widgets](#default-widgets)
  - [Widget Types \& Live Data Resolvers](#widget-types--live-data-resolvers)
  - [API Reference](#api-reference)
    - [Authentication](#authentication-1)
    - [Widget Management](#widget-management)
    - [Dashboard Utilities](#dashboard-utilities)
    - [Model Card](#model-card)
    - [HTML Pages (SPA fallback)](#html-pages-spa-fallback)
  - [Real‑Time Updates (WebSocket)](#realtime-updates-websocket)
  - [Launch Configuration](#launch-configuration)
  - [Model Card Generator](#model-card-generator)
  - [Middleware](#middleware)
  - [Admin Panel](#admin-panel)
  - [Integration with Frontend](#integration-with-frontend)
  - [Management Commands](#management-commands)
    - [`init_widgets`](#init_widgets)
  - [Signals](#signals)
  - [Settings Reference](#settings-reference)
  - [Getting Started](#getting-started)

---

## Architecture

The `web_dashboard` app is the **presentation layer** of the platform. It does **not** contain business logic; instead, it aggregates data from all other apps and exposes it via a **REST API** and **WebSocket**.  

The frontend (a React/Vue SPA) consumes these endpoints to render a fully interactive, real‑time dashboard.

```
┌──────────────────────────────────────────────────────┐
│  Frontend (SPA)                                     │
│  - Login / Register                                  │
│  - Drag & Drop Dashboard                             │
│  - Widget Configuration                              │
│  - Real‑Time Updates (WebSocket)                     │
└──────────────┬───────────────────────────────────────┘
               │ HTTP REST + WebSocket
┌──────────────▼───────────────────────────────────────┐
│  web_dashboard (Django)                              │
│  - Widget CRUD & Layout Persistence                   │
│  - Live Data Resolvers (15+ widget types)             │
│  - Authentication (JWT + cookie)                     │
│  - Launch Config (framework registries)              │
│  - Dashboard Statistics                              │
│  - Model Card Generator                              │
│  - WebSocket Consumer (dashboard_update)             │
└──────────────┬───────────────────────────────────────┘
               │ queries
┌──────────────▼───────────────────────────────────────┐
│  Platform Apps                                       │
│  experiment │ jobscheduler │ sweeps │ terminal       │
│  model_versioning │ notes │ framework                │
└──────────────────────────────────────────────────────┘
```

---

## Features at a Glance

- **Drag‑and‑drop dashboard** – widget positions, sizes, and visibility persist per user.
- **15+ built‑in widget types** – recent runs, GPU monitor, loss/accuracy charts, notes feed, scheduler queue, system health, sweep monitor, terminal, live log, etc.
- **Live data resolvers** – every widget fetches real‑time data from the appropriate app.
- **JWT authentication** – email‑less registration and login, token stored in `authToken` cookie.
- **WebSocket real‑time updates** – dashboard widgets receive live status changes via `DashboardConsumer`.
- **Launch configuration endpoint** – provides all available options (attention types, methods, models, tasks, etc.) from the framework’s registries.
- **Model card generator** – auto‑generates a HuggingFace‑compatible Markdown model card from a training run.
- **Admin panel** – widget administration, dashboard preferences.
- **Middleware** – token authentication for page views, `.well‑known` ignorer.
- **Signals** – broadcasts `TrainingRun` status changes to the WebSocket group.

---

## Models

### `DashboardWidget`

A single tile on the dashboard. Each user can have any number of widgets, arranged on a grid.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `user` | FK → User | Owner of this widget |
| `widget_type` | Char(30) | One of 15 predefined types (see below) |
| `title` | Char(100) | Optional custom title |
| `config` | JSON | Arbitrary configuration (e.g., `run_ids`, `metric`) |
| `position_x` | Int | Column position in grid |
| `position_y` | Int | Row position in grid |
| `width` | Int | Number of grid columns |
| `height` | Int | Number of grid rows |
| `is_visible` | Bool | Whether the widget is displayed |
| `created_at` | DateTime | Auto‑timestamp |

---

## Authentication

The dashboard uses **DRF’s `Token` model** with custom cookie‑based delivery.  

### Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/register/` | Create user, return token, set cookie, log in. |
| `POST` | `/api/login/` | Authenticate user, return token, set cookie. |
| `POST` | `/api/logout/` | Delete token, clear cookie, log out. |

After login, the `authToken` cookie is set with a 30‑day expiry and `SameSite=Lax` for compatibility.  
The frontend can either read the cookie directly or store the token returned in the JSON response.

**Security notes:**
- Passwords are hashed by Django’s `User.create_user`.
- Tokens are deleted on logout.
- The `TokenAuthMiddleware` (see Middleware section) protects all non‑API page views.

---

## Widget System

### Widget CRUD

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/widgets/` | List all widgets for current user. If none exist, create 6 defaults. |
| `PUT` | `/api/widgets/` | Save the entire dashboard layout (creates, updates, deletes). |

The `PUT /api/widgets/` endpoint expects a JSON body:

```json
{
  "widgets": [
    {"id": "...", "position_x": 0, "position_y": 0, "width": 2, "height": 1},
    {"widget_type": "gpu_monitor", "position_x": 2, "position_y": 0, "width": 1, "height": 1, "title": "GPU"}
  ]
}
```

- If an `id` is present and belongs to the user, that widget is **updated**.
- If no `id` is present, a **new widget** is created.
- Any existing widget not in the list is **deleted**.

### Widget Configuration

Each widget can store a `config` JSON object. For example, a `loss_chart` widget might store:

```json
{"run_ids": ["<uuid1>", "<uuid2>"], "limit": 20}
```

The `GET /api/widgets/<widget_id>/config/` endpoint returns this config for editing.

### Default Widgets

When a new user is created, 6 widgets are automatically provisioned:

| Type | Position |
|------|----------|
| `recent_runs` | (0,0) |
| `gpu_monitor` | (2,0) |
| `run_status` | (0,1) |
| `notes_feed` | (1,1) |
| `scheduler_queue` | (3,0) |
| `system_health` | (3,1) |

---

## Widget Types & Live Data Resolvers

Every widget type has a **data resolver** in `resolve_widget_data()`, which queries the appropriate app and returns a structured JSON response.  
The frontend can poll `/api/widgets/<widget_id>/data/` at a configurable interval or use WebSocket for push updates.

| Widget Type | Data Source | Description |
|-------------|-------------|-------------|
| `recent_runs` | `TrainingRun` | Latest training runs with progress. Config: `limit`. |
| `gpu_monitor` | `torch.cuda` | GPU count, memory. |
| `loss_chart` | `TrainingSnapshot` | Loss curves for specified runs. Config: `run_ids`. |
| `accuracy_chart` | `TrainingSnapshot` | Accuracy curves. Config: `run_ids`. |
| `metric_comparison` | `TrainingRun` | Compare a specific metric across runs. Config: `run_ids`, `metric`. |
| `run_status` | `TrainingRun` | Counts by status (running, completed, failed, queued). |
| `notes_feed` | `ResearcherNote` | Latest public notes. Config: `limit`. |
| `model_registry` | `ModelVersion` | Recently published models. Config: `limit`. |
| `scheduler_queue` | `Job` + scheduler | Queued jobs, GPU availability. |
| `custom_plot` | – | Static Plotly JSON stored in config. |
| `terminal` | – | Returns the WebSocket endpoint `/ws/terminal/`. |
| `sweep_monitor` | `Sweep` + `SweepTrial` | Best value, trial list. Config: `sweep_id`. |
| `experiment_comparison` | `TrainingRun` | Multi‑metric comparison across runs. Config: `run_ids`. |
| `system_health` | `psutil` | CPU, memory, disk usage. |
| `live_log` | `TrainingRun.logs` | Last 5000 characters of a run’s log. Config: `run_ids`. |
| `recently_viewed` | `TrainingRun` | Last 5 completed runs (placeholder). |
| `quick_actions` | – | Static links to key platform pages. |
| `system_status` | `TrainingRun`, `User` | Last failed run, active users. |

The resolvers are designed to be **extensible** – adding a new widget type only requires adding a new `elif` branch in `resolve_widget_data()`.

---

## API Reference

### Authentication

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `POST` | `/api/register/` | `AllowAny` | Register & get token |
| `POST` | `/api/login/` | `AllowAny` | Login & get token |
| `POST` | `/api/logout/` | `IsAuthenticated` | Logout & clear token |

### Widget Management

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `GET` | `/api/widgets/` | `IsAuthenticated` | List user’s widgets |
| `PUT` | `/api/widgets/` | `IsAuthenticated` | Save layout (create/update/delete) |
| `GET` | `/api/widgets/<uuid:widget_id>/data/` | `IsAuthenticated` | Live data for a widget |
| `GET` | `/api/widgets/<uuid:widget_id>/config/` | `IsAuthenticated` | Get widget config |

### Dashboard Utilities

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `GET` | `/api/dashboard/stats/` | `IsAuthenticated` | Aggregated stats (runs, sweeps, jobs, models, notes) |
| `GET` | `/api/dashboard/settings/` | `IsAuthenticated` | Get user dashboard preferences |
| `PUT` | `/api/dashboard/settings/` | `IsAuthenticated` | Update dashboard preferences |
| `GET` | `/api/config/all/` | `IsAuthenticated` | All launch options (attentions, methods, tasks, models, etc.) |

### Model Card

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `GET` | `/api/runs/<uuid:run_id>/model-card/` | `IsAuthenticated` | Generate HF model card Markdown |

### HTML Pages (SPA fallback)

| URL | Description |
|-----|-------------|
| `/` | Recent runs listing |
| `/dashboard/` | Full drag‑and‑drop dashboard |
| `/dashboard/job/<uuid:job_id>/` | Experiment monitor detail |
| `/login/` | Login page |
| `/register/` | Registration page |

---

## Real‑Time Updates (WebSocket)

The `DashboardConsumer` (at `ws://<host>:<port>/ws/dashboard/`) allows pushing live updates to the frontend.

- Each authenticated user joins a group `dashboard_<user_id>`.
- When a `TrainingRun` is saved, a **signal** broadcasts the update to this group.
- The frontend can listen for events and refresh the relevant widgets without polling.

The consumer sends JSON messages of the form:

```json
{
  "id": "<run-uuid>",
  "status": "running",
  "name": "my_experiment"
}
```

---

## Launch Configuration

The `GET /api/config/all/` endpoint returns a comprehensive JSON object with **every available option** for the job creation wizard.  
It reads directly from the framework registries:

```json
{
  "attentions": ["full", "galerkin", "linformer", ...],
  "compressors": ["aasvd", "kmeans", ...],
  "methods": ["baseline", "lora", "dora", ...],
  "losses": ["cross_entropy", "focal", ...],
  "optimizers": ["adamw", "sgd", "lamb", ...],
  "schedulers": ["linear", "cosine", ...],
  "callbacks": ["memory", "time", ...],
  "augmentations": ["random_mask", ...],
  "tasks": ["cola", "sst2", "mrpc", ...],
  "models": ["distilbert-base-uncased", "bert-base-uncased", ...],
  "distributed_strategies": ["none", "ddp", "deepspeed", "fsdp"],
  "model_variants": ["standard", "unsloth", "compressed", "hybrid_svd"],
  "tracking_backends": ["local", "tensorboard", "wandb", "mlflow"],
  "storage_backends": ["local", "s3", "gcp", "azure", "hf_hub"]
}
```

This allows the frontend to populate dropdowns dynamically without hard‑coding any option.

---

## Model Card Generator

`GET /api/runs/<run_id>/model-card/` produces a Markdown model card that follows the **HuggingFace model card template**.  
It includes:

- YAML frontmatter with metadata (language, license, tags, metrics).
- Hyperparameter table (learning rate, batch size, epochs, rank, etc.).
- Evaluation results table.
- Usage example.

This card can be pushed to the Hub alongside the model.

---

## Middleware

Two custom middlewares are provided:

- **`TokenAuthMiddleware`** – protects all HTML page views (not `/api/`, `/login/`, `/register/`, `/static/`). If the `authToken` cookie is missing or invalid, redirects to the login page.
- **`WellKnownIgnoreMiddleware`** – returns 204 for requests to `/.well-known/`, preventing browser dev‑tools errors.

To activate, add to `MIDDLEWARE` in `settings.py`:

```python
MIDDLEWARE = [
    ...
    'web_dashboard.middleware.TokenAuthMiddleware',
    'web_dashboard.middleware.WellKnownIgnoreMiddleware',
]
```

---

## Admin Panel

The `DashboardWidget` model is registered in Django admin. You can view, filter, and edit widgets at `/admin/web_dashboard/dashboardwidget/`.

---

## Integration with Frontend

The frontend should:

1. **Auth flow**: Call `POST /api/login/` → store the returned token or read the `authToken` cookie.
2. **Load dashboard**: Call `GET /api/widgets/` → render a grid of widgets.
3. **Live data**: Poll `/api/widgets/<id>/data/` every N seconds, or open a WebSocket at `/ws/dashboard/` for push updates.
4. **Layout persistence**: On drag/resize, call `PUT /api/widgets/` with the full widget list.
5. **Create job wizard**: Fetch `GET /api/config/all/` to populate all dropdowns, then submit to `POST /api/jobs/` (from `jobscheduler`).

---

## Management Commands

### `init_widgets`

```bash
python manage.py init_widgets
```

Creates the default set of 6 widgets for every existing user. Useful after deployment or when adding new default widgets.

---

## Signals

The `signals.py` file connects to `TrainingRun`’s `post_save` signal.  
Whenever a `TrainingRun` is created or updated, a message is sent to the `dashboard_<user_id>` WebSocket group, enabling real‑time UI updates.

---

## Settings Reference

You can store user‑specific dashboard preferences (e.g., theme, refresh interval) using the `/api/dashboard/settings/` endpoint.  
The user model must have a `dashboard_preferences` JSON field (add via migration if needed).  
The API returns these preferences on `GET` and saves them on `PUT`.

---

## Getting Started

1. **Add the app** to `INSTALLED_APPS`:
   ```python
   INSTALLED_APPS += ['web_dashboard']
   ```

2. **Run migrations**:
   ```bash
   python manage.py makemigrations web_dashboard
   python manage.py migrate
   ```

3. **Add middleware** (optional):
   ```python
   MIDDLEWARE += [
       'web_dashboard.middleware.TokenAuthMiddleware',
       'web_dashboard.middleware.WellKnownIgnoreMiddleware',
   ]
   ```

4. **Add URL routing**:
   ```python
   urlpatterns += [path('', include('web_dashboard.urls'))]
   ```

5. **Start server**:
   ```bash
   python manage.py runserver
   ```

6. **Open** `http://localhost:8000/dashboard/` – login, and you’ll see the default widgets.

