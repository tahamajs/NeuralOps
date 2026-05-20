# `sweeps` – Complete Hyperparameter Optimization App

A **production‑grade hyperparameter optimization (HPO)** Django application built on **Optuna**.  
It supports **8+ search algorithms**, **4 pruning schedulers**, **multi‑objective optimization**, **conditional search spaces**, **constraints**, **preferential optimization**, **warm‑starting**, **rich visualizations**, **artifact storage**, **export/import**, and **deep integration** with the existing `experiment.TrainingRun` model.  
Designed to be a drop‑in app that powers the sweep dashboard of the NeuralOps platform.

---

## Table of Contents

- [`sweeps` – Complete Hyperparameter Optimization App](#sweeps--complete-hyperparameter-optimization-app)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Models](#models)
    - [`Sweep`](#sweep)
    - [`SweepTrial`](#sweeptrial)
  - [API Reference](#api-reference)
  - [Search Algorithms](#search-algorithms)
    - [Sampler‑specific options](#samplerspecific-options)
  - [Pruning Schedulers](#pruning-schedulers)
    - [How pruning works](#how-pruning-works)
  - [Multi‑Objective Optimization](#multiobjective-optimization)
  - [Conditional Search Spaces \& Constraints](#conditional-search-spaces--constraints)
    - [Conditional parameters](#conditional-parameters)
    - [Constraints (GP sampler)](#constraints-gp-sampler)
  - [Preferential Optimization](#preferential-optimization)
  - [Warm‑Start \& Resume](#warmstart--resume)
  - [Intermediate Reporting \& Pruning](#intermediate-reporting--pruning)
  - [Visualizations](#visualizations)
  - [Export \& Import](#export--import)
  - [Trial Management](#trial-management)
  - [Task Queue](#task-queue)
  - [Admin Panel](#admin-panel)
  - [Configuration](#configuration)
  - [Getting Started](#getting-started)

---

## Architecture

The `sweeps` app is a **Celery‑powered HPO engine** that uses **Optuna** for intelligent parameter suggestion and pruning.  
A `Sweep` model defines the optimization study, and `SweepTrial` instances represent individual training runs that are executed as `TrainingRun` records in the `experiment` app.

The flow:

1. User creates a `Sweep` via the REST API, providing search space, algorithm, pruner, and base training config.
2. A Celery task (`start_sweep`) starts a loop that:
   - Calls **Optuna** to suggest a new set of hyperparameters.
   - Creates a `SweepTrial` and a corresponding `TrainingRun`.
   - Submits the training job to the job scheduler (`experiment.tasks.train_job`).
   - Waits for parallel worker slots.
   - When a trial finishes, a **callback** task (`report_sweep_result`) reports the metric back to Optuna and updates the trial record.
3. The loop runs until `max_trials` is reached, the sweep is paused, or too many consecutive failures occur.

During execution, the engine supports **intermediate reporting** for pruning, and the API provides **live visualizations** and trial management.

---

## Models

### `Sweep`

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `project` | FK → `experiment.Experiment` | Parent project |
| `name` | Char(255) | Human‑readable name |
| `description` | Text | Optional description |
| `config` | JSON | Full configuration: search space, base_config, constraints, etc. |
| `status` | Enum | `active`, `paused`, `completed`, `failed` |
| `direction` | Enum | `maximize` / `minimize` (for single‑objective) |
| `directions` | JSON | List of `"maximize"`/`"minimize"` per objective (multi‑objective) |
| `metric_name` | Char(100) | Name of primary metric (e.g., `eval_accuracy`) |
| `metric_names` | JSON | List of metric names for multi‑objective |
| `algorithm` | Char(50) | Optuna sampler: `tpe`, `cmaes`, `nsgaii`, `nsgaiii`, `grid`, `random`, `qmc`, `gp`, `botorch`, `auto` |
| `pruner` | Char(50) | Optuna pruner: `hyperband`, `median`, `wilcoxon`, `successive_halving`, `none` |
| `scheduler_type` | Char(50) | Trial scheduling: `fifo`, `asha`, `pbt`, `pb2`, `none` (currently used for planning) |
| `max_trials` | Int | Maximum number of trials to run |
| `max_failures` | Int | Consecutive failures before marking sweep failed |
| `parallel_workers` | Int | Maximum concurrent trials |
| `warm_start_from` | FK → self | Resume / warm‑start from a previous sweep |
| `preference_enabled` | Bool | Enable preferential optimization (human‑in‑the‑loop) |
| `preference_queries` | JSON | Stored preference records |
| `artifact_dir` | Char(500) | Directory for storing trial artifacts |
| `tags` | JSON | List of tags |
| `created_at` / `updated_at` | DateTime | Timestamps |

### `SweepTrial`

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `sweep` | FK → Sweep | Parent sweep |
| `trial_number` | Int | Trial number (from Optuna) |
| `status` | Enum | `queued`, `running`, `completed`, `failed`, `pruned` |
| `objective_value` | Float | Primary objective value (for single‑objective) |
| `objectives` | JSON | List of objective values (for multi‑objective) |
| `params` | JSON | Hyperparameter values used |
| `user_attrs` | JSON | Arbitrary user attributes (e.g., notes, environment) |
| `notes` | Text | Human notes |
| `is_pareto` | Bool | Whether this trial is on the Pareto front |
| `run` | OneToOne → `TrainingRun` | Linked training run |
| `artifacts` | JSON | List of artifact paths/URLs |
| `duration_seconds` | Float | Wall time of the trial |
| `created_at` | DateTime | When the trial was created |

---

## API Reference

All endpoints are prefixed with `/api/sweeps/` and require authentication (`IsAuthenticated`).

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/sweeps/` | List all sweeps |
| `POST` | `/sweeps/` | Create a new sweep (starts execution via Celery) |
| `GET` | `/sweeps/<uuid:sweep_id>/` | Get sweep detail |
| `PATCH` | `/sweeps/<uuid:sweep_id>/` | Update sweep status (`active`/`paused`) |
| `GET` | `/sweeps/<uuid:sweep_id>/trials/` | List all trials of a sweep |
| `GET` | `/sweeps/<uuid:sweep_id>/trials/<int:trial_number>/` | Get trial detail |
| `PATCH` | `/sweeps/<uuid:sweep_id>/trials/<int:trial_number>/update/` | Update trial notes / user attrs |
| `POST` | `/sweeps/<uuid:sweep_id>/trials/<int:trial_number>/intermediate/` | Report an intermediate value (for pruning) |
| `POST` | `/sweeps/<uuid:sweep_id>/preference/` | Submit human preference between two trials |
| `GET` | `/sweeps/<uuid:sweep_id>/importance/` | Parameter importance (Optuna) |
| `GET` | `/sweeps/<uuid:sweep_id>/pareto/` | Pareto front data (multi‑objective) |
| `GET` | `/sweeps/<uuid:sweep_id>/contour/?params=lr,batch` | Contour plot |
| `GET` | `/sweeps/<uuid:sweep_id>/history/` | Optimization history |
| `GET` | `/sweeps/<uuid:sweep_id>/slice/?param=lr` | Slice plot |
| `GET` | `/sweeps/<uuid:sweep_id>/parallel/` | Parallel coordinates plot |
| `GET` | `/sweeps/<uuid:sweep_id>/edf/` | Empirical attainment surface (multi‑objective) |
| `GET` | `/sweeps/<uuid:sweep_id>/warm-start/` | Get warm‑start config |
| `GET` | `/sweeps/<uuid:sweep_id>/dataframe/` | Trials as dataframe (JSON) |
| `GET` | `/sweeps/<uuid:sweep_id>/export/` | Export study to JSON file |
| `POST` | `/sweeps/import/` | Import trials from a JSON file into a study |

---

## Search Algorithms

The app supports all major Optuna samplers. You select one via `algorithm` in the sweep config.

| Algorithm | Key | Description |
|-----------|-----|-------------|
| **TPE** (Tree‑structured Parzen Estimator) | `tpe` | Default; excellent for categorical and conditional spaces. |
| **CMA‑ES** (Covariance Matrix Adaptation Evolution Strategy) | `cmaes` | Good for continuous, high‑dimensional spaces. |
| **NSGA‑II** | `nsgaii` | Multi‑objective genetic algorithm. |
| **NSGA‑III** | `nsgaiii` | Multi‑objective with reference points. |
| **Grid Search** | `grid` | Exhaustive search over a grid. |
| **Random Search** | `random` | Simple random sampling. |
| **Quasi‑Monte Carlo (QMC)** | `qmc` | Better than random for continuous spaces. |
| **GP (Gaussian Process)** | `gp` | Bayesian optimization with constraints support. |
| **BoTorch** | `botorch` | Advanced Bayesian optimization (PyTorch). |
| **Auto Sampler** | `auto` | Optuna chooses the best sampler automatically. |

### Sampler‑specific options

- For `gp`, you can specify `constraints` in the sweep config to enforce feasibility.
- For `nsgaii` and `nsgaiii`, `directions` must contain multiple objectives.

---

## Pruning Schedulers

Pruners stop unpromising trials early, saving resources. Set `pruner` in the sweep config.

| Pruner | Key | Description |
|--------|-----|-------------|
| **Hyperband** | `hyperband` | Adaptive resource allocation; stops poor trials early. |
| **Median** | `median` | Prunes trials below the median of previous trials at the same step. |
| **Wilcoxon** | `wilcoxon` | Uses the Wilcoxon signed‑rank test to decide pruning. |
| **Successive Halving** | `successive_halving` | Uniformly allocates a fixed budget across trials. |
| **No Pruner** | `none` | No pruning (runs every trial to completion). |

### How pruning works

During training, the training code should periodically call the `/intermediate/` endpoint with step and metric value.  
The engine then asks Optuna whether the trial should be pruned. If yes, the trial is marked `pruned` and no further training occurs.

---

## Multi‑Objective Optimization

When multiple metrics are specified in `metric_names` and `directions`, the sweep performs multi‑objective optimization.

- **Directions**: Each metric must have a direction (`maximize` or `minimize`).
- The engine passes the list of directions to Optuna’s `create_study`.
- The Pareto front is computed automatically, and trials belonging to it are flagged (`is_pareto=True`).
- You can retrieve the Pareto front via the `/pareto/` endpoint.
- Visualization endpoints (`/edf/`, `/pareto/`) are available only for multi‑objective sweeps.

---

## Conditional Search Spaces & Constraints

The search space supports **conditional parameters** and **constraints**.

### Conditional parameters

In the search space definition, a parameter can have a `condition` field:

```json
{
  "optimizer": {
    "type": "categorical",
    "choices": ["adamw", "sgd"]
  },
  "momentum": {
    "type": "float",
    "low": 0.8,
    "high": 0.999,
    "condition": {"param": "optimizer", "value": "sgd"}
  }
}
```

Here, `momentum` is only suggested when `optimizer` equals `"sgd"`.

### Constraints (GP sampler)

When using the `gp` sampler, you can define constraints in the sweep config:

```json
{
  "algorithm": "gp",
  "constraints": [
    {"param": "batch_size", "threshold": 128, "op": "le"}
  ]
}
```

The constraint function returns a value ≤ 0 if the trial is feasible. Optuna will avoid suggesting infeasible trials.

---

## Preferential Optimization

When `preference_enabled=True`, human feedback can be incorporated into the optimization process.  
This is useful when the objective function is hard to quantify (e.g., “quality of generated text”).

- The `/preference/` endpoint accepts a pair of trials (`trial_a`, `trial_b`) and a preference (`"a"` or `"b"`).
- Preferences are stored in the sweep and can later be used with a custom preferential sampler or a plugin (not directly in this app, but the data is ready for integration).

---

## Warm‑Start & Resume

Set `warm_start_from` to the ID of a previous sweep to reuse its history.  
The engine will initialize the new Optuna study with the old trials, effectively continuing the optimization from where it left off.

- The `/warm-start/` endpoint returns a config snippet containing the best parameters and value from a sweep, useful for seeding new sweeps.
- Pausing and resuming a sweep is done via the `PATCH /sweeps/<id>/` endpoint (set `status` to `active` or `paused`).

---

## Intermediate Reporting & Pruning

To enable pruning, the training process should periodically report intermediate objective values.  
Send a POST request to `/api/sweeps/<sweep_id>/trials/<trial_number>/intermediate/` with:

```json
{
  "step": 100,
  "value": 0.85
}
```

The engine calls `trial.report(value, step)` in Optuna.  
It then checks `trial.should_prune()` and returns `{"pruned": true/false}`.  
If pruned, the trial status is changed to `pruned` and training stops.

---

## Visualizations

The app exposes every major Optuna visualization as JSON (Plotly figure data) via the API.  
These can be rendered directly in the frontend using Plotly.js.

| Endpoint | Optuna function | Description |
|----------|----------------|-------------|
| `/importance/` | `plot_param_importances` | Bar chart of hyperparameter importance |
| `/pareto/` | `get_pareto_front_trials` | List of non‑dominated trials |
| `/contour/?params=lr,batch` | `plot_contour` | Contour plot for two parameters |
| `/history/` | `plot_optimization_history` | Best value vs trial number |
| `/slice/?param=lr` | `plot_slice` | Slice plot for one parameter |
| `/parallel/` | `plot_parallel_coordinate` | Parallel coordinates for all parameters |
| `/edf/` | `plot_edf` | Empirical distribution function (multi‑objective) |

All plot endpoints return JSON suitable for Plotly’s `Plotly.newPlot`.

---

## Export & Import

Sweeps can be exported to a JSON file (containing all trial parameters and values) and imported into another sweep.

- **Export**: `GET /api/sweeps/<id>/export/` downloads a JSON file.
- **Import**: `POST /api/sweeps/import/` with a multipart file upload and `sweep_id`.

This is useful for sharing optimization results or transferring studies between environments.

---

## Trial Management

- Each trial can have `notes` and `user_attrs` (arbitrary JSON) updated via the `/update/` endpoint.
- `user_attrs` are also synced with Optuna trial user attributes.
- The `is_pareto` flag is automatically set after each trial completion for multi‑objective sweeps.
- Artifacts (e.g., saved model checkpoints, plots) can be tracked via the `artifacts` JSON list.

---

## Task Queue

The sweep execution uses Celery:

- **`start_sweep`** – Main loop that runs until the sweep is paused/completed.
- **`report_sweep_result`** – Callback after a training run finishes, reporting metrics back to Optuna and updating the trial record.

Both tasks are idempotent and handle errors gracefully. The main loop enforces `parallel_workers` and `max_failures`.

---

## Admin Panel

Both `Sweep` and `SweepTrial` models are registered with Django admin.  
You can view sweep status, trial values, parameters, and filter by status/algorithm/project.

---

## Configuration

A typical sweep creation payload looks like this:

```json
{
  "project": "<experiment-uuid>",
  "name": "lr_batch_optimization",
  "description": "Find best lr and batch size",
  "config": {
    "base_config": {
      "model_name": "distilbert-base-uncased",
      "task": "sst2",
      "epochs": 3,
      "max_seq_length": 128
    },
    "search_space": {
      "learning_rate": {
        "type": "float",
        "low": 1e-6,
        "high": 1e-3,
        "log": true
      },
      "batch_size": {
        "type": "int",
        "low": 8,
        "high": 64,
        "step": 8
      },
      "optimizer": {
        "type": "categorical",
        "choices": ["adamw", "sgd"]
      },
      "momentum": {
        "type": "float",
        "low": 0.8,
        "high": 0.999,
        "condition": {"param": "optimizer", "value": "sgd"}
      }
    },
    "max_trials": 50,
    "parallel_workers": 2,
    "algorithm": "tpe",
    "pruner": "hyperband",
    "direction": "maximize",
    "metric_name": "eval_accuracy"
  }
}
```

All top‑level fields (algorithm, pruner, direction, etc.) can also be set directly on the Sweep model when using `SweepCreateSerializer`.

---

## Getting Started

1. **Install the app**: Add `'sweeps'` to `INSTALLED_APPS` and run migrations.
2. **Ensure Celery** is configured and running with a broker (Redis/RabbitMQ).
3. **Create a project** (`experiment.Experiment`) to associate the sweep.
4. **Create a sweep** via `POST /api/sweeps/` – the Celery task `start_sweep` will be queued.
5. **Monitor progress** via the sweeps list, trial list, and visualization endpoints.
6. **Adjust** – pause/resume sweeps, submit preferences, export results.

