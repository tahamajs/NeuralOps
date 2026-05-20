# Model Versioning

Complete model artifact management with compression, diffing, cold storage, and HuggingFace Hub publishing.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Services](#services)
- [Signals & Events](#signals--events)
- [REST API](#rest-api)
- [Admin Interface](#admin-interface)
- [Usage Examples](#usage-examples)
- [Testing](#testing)

---

## Overview

The `model_versioning` app provides a **full lifecycle** for every trained model:

- **Automatic version creation** when a training run completes.
- **Compression** using our framework compressors (A3, K‑Means, TT, etc.) or generic gzip.
- **Diff generation** between consecutive versions to save storage.
- **Storage tier management** – hot (local), warm (cloud), cold (archive).
- **HuggingFace Hub publishing** – one‑click upload with auto‑generated model cards.
- **Production flag** – mark the best version as the production model.
- **Cleanup** – delete old versions while keeping the best and tagged ones.

The system is event‑driven: every action (create, publish, archive, set production) emits a signal, logs an activity, sends an email, and optionally triggers a webhook.

---

## Models

### ModelVersion

The core model. Each row represents one version of a model produced by a training run (`Job`).

| Field | Type | Description |
|-------|------|-------------|
| `run` | FK → `Job` | The training run that produced this version. |
| `version` | char | Human‑readable version string (`v1.0`, `v2`, …). |
| `version_number` | int | Auto‑incrementing integer unique per run. |
| `artifact_path` | char | Directory containing the uncompressed model files. |
| `compressed_artifact_path` | char | Path to a compressed archive (`.tar.gz`) or a compressed model directory. |
| `compression_method` | char | `'none'`, `'gzip'`, or a framework compressor key (`'a3'`, `'kmeans'`, …). |
| `compression_config` | JSON | Parameters passed to the compressor (e.g., `{"rank": 64}`). |
| `artifact_size_mb` | float | Size of the uncompressed artifact. |
| `compressed_size_mb` | float | Size of the compressed artifact (if any). |
| `storage_tier` | choice | `'hot'` (local), `'warm'` (cloud), `'cold'` (archive). |
| `is_published_to_hub` | bool | Whether the model has been pushed to HuggingFace. |
| `hub_repo_url` | URL | HuggingFace repo URL after publishing. |
| `metrics` | JSON | All evaluation metrics (`eval_accuracy`, `eval_loss`, …). |
| `is_production` | bool | If `True`, this version is the current production model for the run. |
| `validation_passed` | bool | Whether the model passed a validation gate. |
| `git_commit` | char | Git commit hash when training started. |
| `config_snapshot_id` | UUID | Link to a saved configuration snapshot. |

### ModelVersionDiff

Stores the **delta** between two versions, allowing reconstruction without storing both full models.

| Field | Type | Description |
|-------|------|-------------|
| `from_version` | FK | The base version. |
| `to_version` | FK | The version derived from the base. |
| `diff_artifact` | char | Path to a `.tar.gz` file containing `diff.pt` (weight differences). |
| `diff_size_mb` | float | Size of the diff archive. |
| `original_size_saved_mb` | float | How much storage was saved vs. storing the full `to_version`. |

### ModelLineage

Tracks **parent‑child relationships** between versions, forming a directed acyclic graph.

| Field | Type | Description |
|-------|------|-------------|
| `parent` | FK | Ancestor version (nullable for initial version). |
| `child` | FK | Descendant version. |
| `relationship_type` | choice | `'fine_tuned'`, `'compressed'`, `'quantized'`, `'distilled'`, `'merged'`, `'continued'`. |
| `notes` | text | Free‑form notes. |

**Graph traversal** is provided by `get_lineage_graph(version_id, direction, max_depth)` which returns `{'nodes': […], 'edges': […]}` ready for a frontend D3.js chart.

---

## Services

The `services.py` module contains the business logic.

### `create_version(run, metrics, artifact_dir=None, compression='gzip', compression_config=None)`

Creates a new `ModelVersion`.  
- Automatically determines the next `version_number`.  
- Optionally compresses the artifact using `'gzip'` or any framework compressor.  
- Returns the new `ModelVersion` instance.

### `load_model(version)`

Loads a HuggingFace model from a version, transparently decompressing if the artifact is stored in gzip or framework‑compressed format.

### `create_diff(from_version, to_version)`

Computes `state_dict` differences between two versions and saves them as a compressed archive.  
Returns a `ModelVersionDiff` instance.

### `load_version_from_diff(base_version, diff)`

Applies a diff to a base model to reconstruct the derived version.

### `auto_select_compression(metrics, size_mb)`

Heuristic to choose the best compression method based on accuracy and model size.

### `migrate_storage_tier(version)`

Automatically moves a version to colder storage based on its age:
- After 30 days → `'warm'` (S3 Standard‑IA).  
- After 90 days → `'cold'` (S3 Glacier).

---

## Signals & Events

All signals are defined in `signals.py`. They are automatically connected when the app starts (via `apps.py`).

| Signal | Sent when | Side effects |
|--------|-----------|--------------|
| `version_created` | A new version is saved. | Activity log, email, webhook. |
| `version_published` | A version is published to HuggingFace. | Activity log, email, webhook. |
| `version_archived` | A version is moved to cold storage. | Activity log, webhook. |
| `version_restored` | A version is restored from cold storage. | Activity log, webhook. |
| `version_set_production` | A version is marked as production. | Activity log, email, webhook. |
| `version_diff_created` | A diff between two versions is created. | Activity log, webhook. |

Additionally, the `post_save` signal on `ModelVersion` triggers `migrate_storage_tier` automatically.

---

## REST API

All endpoints require authentication (`IsAuthenticated`). Staff‑only endpoints are marked.

### Global

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/versions/` | Recent 50 versions across all runs. |
| `GET` | `/api/compression-methods/` | Available compression methods. |
| `POST` | `/api/versions/migrate-lifecycle/` | Run tier migration on all versions (staff). |

### Per‑Run

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/runs/{run_id}/versions/` | Versions for a run (paginated, filterable, CSV). |
| `POST` | `/api/runs/{run_id}/versions/cleanup/` | Delete old versions (staff). |
| `POST` | `/api/runs/{run_id}/versions/{n}/set_production/` | Set production version (staff). |
| `GET` | `/api/runs/{run_id}/versions/compare/{a}/{b}/` | Compare metrics of two versions. |

### Single Version

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/versions/{id}/` | Version detail. |
| `GET` | `/api/versions/{id}/download/` | Download model artifact (prefers compressed). |
| `POST` | `/api/versions/{id}/publish/` | Publish to HuggingFace Hub (staff). |
| `POST` | `/api/versions/{id}/archive/` | Archive to cold storage (staff). |
| `POST` | `/api/versions/{id}/restore/` | Restore from cold storage (staff). |
| `GET` | `/api/versions/{id}/lineage/` | Lineage graph. |

### Diffs

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/versions/diff/{from}/{to}/` | Create a diff between two versions. |
| `GET` | `/api/versions/reconstruct/{base}/{diff}/` | Reconstruct a version from a diff and download. |

---

## Admin Interface

Registered models: `ModelVersion`, `ModelVersionDiff`, `ModelLineage`.

- `ModelVersionAdmin` shows inlines for diffs and lineage edges.  
- All models are searchable and filterable.  
- Read‑only fields for timestamps and IDs.

---

## Usage Examples

### 1. Create a version with compression

```python
from model_versioning.services import create_version

version = create_version(
    run=training_run,
    metrics={"eval_accuracy": 0.92, "eval_loss": 0.35},
    artifact_dir="/media/runs/abc/model",
    compression='a3',
    compression_config={"a3_rank": 64}
)
print(version.compressed_size_mb)   # e.g., 34.2
```

### 2. Create a diff between versions

```python
from model_versioning.services import create_diff

diff = create_diff(v1, v2)
# storage saved: diff.original_size_saved_mb
```

### 3. Publish to HuggingFace Hub

```python
from model_versioning.views import publish_to_hub
# or via API:
# POST /api/versions/{id}/publish/
```

### 4. Download a version

The download endpoint serves the compressed artifact if available, otherwise streams the artifact directory as a `.tar.gz`.

```bash
curl -H "Authorization: Token $TOKEN" http://localhost:8000/api/versions/{id}/download/ -o model.tar.gz
```

### 5. Automatic versioning after training

In the training runner’s `final_callback`:

```python
from model_versioning.services import create_version, auto_select_compression

def on_training_complete(run, metrics, artifact_dir):
    method, config = auto_select_compression(metrics, os.path.getsize(artifact_dir) / 1e6)
    version = create_version(run, metrics, artifact_dir=artifact_dir,
                             compression=method, compression_config=config)
    # Optionally create a diff from the previous best version
    prev = run.model_versions.filter(is_production=True).first()
    if prev and prev.id != version.id:
        create_diff(prev, version)
```

---

## Testing

The test suite (`tests.py`) covers:

- Model creation and constraints
- List filtering, pagination, ordering, CSV export
- Detail, download, publish, archive, restore
- Staff permission enforcement
- Version comparison
- Diff creation and reconstruction
- Lineage graph

Run with:

```bash
python manage.py test model_versioning
```

---

## Dependencies

- `django`, `djangorestframework`
- `huggingface_hub` (for publishing)
- `boto3` (for S3 cold storage)
- `torch`, `transformers`
- The framework compressor registry (optional, falls back gracefully)

