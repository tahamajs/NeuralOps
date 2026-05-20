# 🧠 NeuralOps Research Notebook

A complete, collaborative, and shareable electronic lab notebook (ELN) for machine learning research, built with Django REST Framework, PostgreSQL, Channels, and HuggingFace Hub.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Models](#models)
5. [API Reference](#api-reference)
6. [Real‑time Collaboration](#real‑time-collaboration)
7. [Sharing & Publishing](#sharing--publishing)
8. [Auto‑Generated Reports](#auto‑generated-reports)
9. [Configuration](#configuration)
10. [Deployment](#deployment)
11. [Testing](#testing)

---

## Overview

The **NeuralOps Research Notebook** replaces scattered markdown files and unorganized experiment logs with a structured, API‑first notebook built directly into your ML experiment tracker. Every note is versioned, searchable, and can be shared with your team via public links or published to the HuggingFace Hub as a free, hosted web page. It integrates deeply with the NeuralOps training runs, allowing you to auto‑generate rich experiment reports with charts, hyperparameters, and metrics directly from your training jobs.

---

## Features

- 📝 **Full CRUD** for notes with rich Markdown + LaTeX support
- 📌 **Pinning, favorites, and threaded comments**
- 🏷️ **Flexible tagging and filtering**
- 🔄 **Real‑time collaborative editing** via WebSockets (Channels)
- 🔗 **Bidirectional note linking** (knowledge graph)
- 📊 **Auto‑generated experiment reports** from training runs (with loss/accuracy charts)
- 📁 **File artifacts** – upload and attach checkpoints, plots, logs
- 🧪 **Reusable experiment protocols**
- 📤 **Public share links** with optional password, expiration, and view limits
- 🚀 **One‑click publish to HuggingFace** as a Dataset or static Space (free hosting)
- 🔍 **Full‑text search** (PostgreSQL `SearchVector`)
- 📈 **Dashboard & statistics** – usage, tag clouds, recent activity
- 🔔 **Notifications** for mentions, shares, and comments
- 🌐 **Webhook integration** for external systems
- 🔐 **Fine‑grained permissions** – public, private, shared with specific users

---

## Architecture

The notes app is a standard Django application with the following components:

```
notes/
├── models.py          # 15+ models covering notes, versions, comments, etc.
├── serializers.py     # DRF serializers with computed fields
├── views.py           # 40+ endpoints for all CRUD and advanced operations
├── urls.py            # API routes (versioned)
├── consumers.py       # WebSocket consumer for collaborative editing
├── filters.py         # Advanced filtering (django‑filter)
├── report_generator.py # Auto‑report generation from TrainingRun
├── hf_upload.py       # HuggingFace Hub integration
├── templates.py       # Default note templates
├── signals.py         # Auto‑mentions, notifications, webhook triggers
├── webhook_utils.py   # HMAC signing and delivery
├── admin.py           # Admin panel configuration
└── apps.py            # App config (creates default templates)
```

All API endpoints are prefixed with `/api/notes/v1/` and return JSON. Authentication uses the same token system as the rest of the platform (DRF `TokenAuthentication` or cookie fallback).

---

## Models

### Core Model: `ResearcherNote`

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `author` | FK → User | Creator of the note |
| `run` | FK → Job (nullable) | Linked training run |
| `linked_runs` | M2M → Job | Additional runs referenced in the note |
| `parent_note` | FK → self | For threaded replies |
| `note_type` | Choice | One of `general`, `experiment_log`, `training_report`, `compression_report`, `literature_review`, `hypothesis`, `meeting_notes`, `code_snippet` |
| `title` | CharField | Optional title |
| `content` | TextField | Markdown content with LaTeX support |
| `tags` | JSONField | List of tags (e.g., `["baseline", "overfitting"]`) |
| `metadata` | JSONField | Flexible metadata (hyperparameters, metrics, charts, model versions, etc.) |
| `is_public` | Boolean | Visible to all users (if not soft‑deleted) |
| `is_pinned` | Boolean | Pin to top of lists |
| `is_favorite` | Boolean | Bookmarked by the author |
| `shared_with` | M2M → User | Users with direct access |
| `reactions` | JSONField | Reaction counts by type (like, important, etc.) |
| `version` | Integer | Auto‑incremented on content change |
| `slug` | SlugField | URL‑friendly identifier |
| `search_vector` | SearchVectorField | PostgreSQL full‑text search index |
| `created_at`, `updated_at` | DateTime | Timestamps |
| `deleted_at` | DateTime (nullable) | Soft‑delete timestamp |

### Supporting Models

- **NoteVersion** – full content snapshot on each edit
- **NoteImage** – embedded images with ordering
- **NoteMention** – tracks @username mentions
- **NoteComment** – threaded comments with resolve state
- **NoteTemplate** – reusable content templates with variable substitution
- **ShareableLink** – public, revocable links with password, expiration, max views
- **Artifact** – file attachments (checkpoints, plots, logs) with size tracking
- **Protocol** – reusable experiment steps
- **NoteLink** – bidirectional links between notes (knowledge graph)
- **WebhookSubscription** – external webhook registration
- **Notification** – user notifications for mentions, shares, comments, etc.

---

## API Reference

All endpoints require authentication (`IsAuthenticated`) except where noted.  
Responses are paginated (20 items per page, configurable).

### Notes CRUD

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notes/` | List notes (paginated, filterable, searchable) |
| `POST` | `/api/notes/v1/notes/` | Create a new note |
| `GET` | `/api/notes/v1/notes/{id}/` | Retrieve a note |
| `PUT/PATCH` | `/api/notes/v1/notes/{id}/` | Update a note (author only) |
| `DELETE` | `/api/notes/v1/notes/{id}/` | Soft‑delete a note |

**Filtering (query parameters):**
- `title`, `content` – case‑insensitive contains
- `tags` – filter by tag (e.g., `?tags=baseline`)
- `note_type` – one of the `NOTE_TYPE` choices
- `is_public`, `is_pinned`, `is_favorite`
- `author_username` – exact username match
- `run_id` – filter notes linked to a specific run
- `created_at_after`, `created_at_before` – ISO 8601 datetime range
- `search` – full‑text search across title and content

**Export:** add `?export=csv` or `?export=json` to bypass pagination.

### Notes – Special Operations

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/notes/v1/notes/{id}/react/` | Toggle a reaction (like, important, etc.) |
| `POST` | `/api/notes/v1/notes/{id}/upload-image/` | Upload an inline image |
| `POST` | `/api/notes/v1/notes/{id}/favorite/` | Toggle favorite status |
| `PATCH` | `/api/notes/v1/notes/{id}/pin/` | Toggle pin status |
| `POST` | `/api/notes/v1/notes/{id}/share/` | Share with a specific user (by username) |
| `POST` | `/api/notes/v1/notes/{id}/unshare/` | Remove a user from shared list |
| `GET` | `/api/notes/v1/notes/shared-with-me/` | Notes shared with the current user |
| `GET` | `/api/notes/v1/notes/runs/{run_id}/` | Notes linked to a training run |

### Version History

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notes/{id}/versions/` | List all versions of a note |
| `GET` | `/api/notes/v1/notes/{id}/versions/{version_number}/` | Retrieve a specific version snapshot |

### Public Share Links

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `POST` | `/api/notes/v1/notes/{id}/share-link/` | Yes | Create a public share link |
| `GET` | `/api/notes/v1/notes/{id}/share-links/` | Yes | List all share links for a note |
| `POST` | `/api/notes/v1/notes/{id}/share-links/{link_id}/revoke/` | Yes | Revoke a share link |
| `GET` | `/api/notes/shared/{token}/` | **No** | View a note via public link (supports `?password=...`) |

**Create share link body:** `{ "expires_in_days": 7, "password": "optional", "max_views": 100 }`  
**Response:** `{ "url": "https://.../notes/shared/abc123/", "token": "abc123", ... }`

### HuggingFace Upload

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/notes/v1/notes/{id}/upload-hf/` | Upload note as a HuggingFace Dataset |
| `POST` | `/api/notes/v1/notes/{id}/upload-hf-space/` | Upload note as a static HTML Space (free hosting) |
| `POST` | `/api/notes/v1/notes/{id}/sync-hf/` | Sync latest content to a previously uploaded HF repo |

### Artifacts (File Attachments)

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notes/{id}/artifacts/` | List artifacts attached to a note |
| `POST` | `/api/notes/v1/notes/{id}/artifacts/` | Upload a new artifact (multipart form) |
| `DELETE` | `/api/notes/v1/notes/{id}/artifacts/{artifact_id}/` | Delete an artifact |
| `GET` | `/api/notes/v1/notes/{id}/artifacts/{artifact_id}/download/` | Download the artifact file |

### Protocols

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/protocols/` | List all protocols (public or own) |
| `POST` | `/api/notes/v1/protocols/` | Create a new protocol |
| `GET` | `/api/notes/v1/protocols/{id}/` | Retrieve a protocol |
| `PUT/PATCH` | `/api/notes/v1/protocols/{id}/` | Update a protocol (owner only) |
| `DELETE` | `/api/notes/v1/protocols/{id}/` | Delete a protocol (owner only) |

### Knowledge Graph (Note Links)

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/notes/v1/notes/links/` | Create a link between two notes |
| `GET` | `/api/notes/v1/notes/{id}/links-graph/` | Get the knowledge graph centered on a note |
| `DELETE` | `/api/notes/v1/notes/links/{link_id}/` | Delete a link |

**Create link body:** `{ "source_id": "...", "target_id": "...", "link_type": "extends" }`  
**Graph response:** `{ "nodes": [...], "edges": [...] }` – ready for D3.js/vis.js.

### Comments

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notes/{id}/comments/` | List all top‑level comments on a note |
| `POST` | `/api/notes/v1/notes/{id}/comments/` | Add a comment (supports `parent_comment_id` for replies) |

### Auto‑Generated Reports

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/notes/v1/notes/generate-report/{run_id}/` | Generate a rich training report from a run |

This creates a new note of type `training_report` with embedded charts, metrics, and configuration details.

### Templates

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/templates/` | List available note templates |
| `POST` | `/api/notes/v1/templates/` | Create a custom template |
| `POST` | `/api/notes/v1/templates/{id}/render/` | Render a template with variables → creates a note |

### Search & Dashboard

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/search/?q=...` | Full‑text search across all notes |
| `GET` | `/api/notes/v1/dashboard/` | Usage statistics and activity feed |

### Notifications

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notifications/` | List user notifications |
| `POST` | `/api/notes/v1/notifications/read-all/` | Mark all notifications as read |

### Bulk Operations

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/notes/v1/notes/bulk/` | Bulk delete or create notes |
| `PATCH` | `/api/notes/v1/notes/bulk/tags/` | Add/remove tags on multiple notes |

### Webhooks

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/webhooks/` | List registered webhooks |
| `POST` | `/api/notes/v1/webhooks/` | Register a new webhook |
| `DELETE` | `/api/notes/v1/webhooks/{id}/` | Delete a webhook |
| `POST` | `/api/notes/v1/webhooks/{id}/test/` | Send a test payload |

### Export

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notes/{id}/export/pdf/` | Export note as PDF (requires `weasyprint`) |
| `GET` | `/api/notes/v1/notes/{id}/export/md/` | Export note as raw Markdown |

---

## Real‑time Collaboration

The app includes a WebSocket consumer (`NoteCollaborationConsumer`) that allows multiple users to edit the same note simultaneously. Changes are broadcast to all connected clients in real time.

**Endpoint:** `ws://<host>/ws/notes/{note_id}/?token=<auth_token>`

The consumer handles:
- `init` – sends the current note content to a new joiner
- `edit` – broadcasts content updates
- `cursor` – broadcasts cursor positions (for showing other users’ cursors)
- `presence` – notifies when users join/leave

---

## Sharing & Publishing

### Direct Sharing
Share a note with specific users via `POST /api/notes/v1/notes/{id}/share/`. Shared users can view the note in their “shared with me” list.

### Public Share Links
Create a revocable, optionally password‑protected link that anyone can access **without authentication**. Links can expire after a set time or after a maximum number of views.

### HuggingFace Hub
Publish a note to the HuggingFace Hub with one click:

- **Dataset** – creates a dataset card with the note rendered as Markdown. Anyone can view it at `https://huggingface.co/datasets/<username>/neuralops-note-<id>`.
- **Static Space** – creates a free static website with a styled HTML version of the note. Viewable at `https://huggingface.co/spaces/<username>/neuralops-note-<id>`.

Both options are **free** and can be made public.

---

## Auto‑Generated Reports

The `ExperimentReportGenerator` class automatically creates a comprehensive Markdown report from a training run. The report includes:

- Model and task configuration
- Final metrics (accuracy, F1, loss) with trend analysis
- Loss/accuracy charts (embedded as base64 PNG)
- Training analysis (overfitting, plateau, divergence detection)
- Linked artifacts and next‑step suggestions

**Usage:** `POST /api/notes/v1/notes/generate-report/{run_id}/` or call `generate_experiment_report(run_id, author)` from your training pipeline.

---

## Configuration

Add the required settings to your Django `settings.py`:

```python
# HuggingFace Hub (optional, for publishing notes)
HUGGINGFACE_TOKEN = "hf_..."
HUGGINGFACE_USERNAME = "your-username"

# Public base URL (used for share links)
PUBLIC_BASE_URL = "https://your-domain.com"

# WebSocket / Channels (for real‑time collaboration)
ASGI_APPLICATION = "your_project.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [("redis", 6379)]},
    },
}
```

Install additional dependencies:

```bash
pip install weasyprint           # PDF export
pip install django-filter       # advanced filtering
pip install channels            # WebSockets
pip install channels-redis      # production channel layer
pip install huggingface-hub     # HF upload
```

---

## Deployment

1. **Run migrations**:
   ```bash
   python manage.py makemigrations notes
   python manage.py migrate
   ```

2. **Collect static files** (if using the admin or serving uploaded files):
   ```bash
   python manage.py collectstatic
   ```

3. **Set up Redis** for Channels and Celery (if not already configured).

4. **Configure `ALLOWED_HOSTS`** and `PUBLIC_BASE_URL` for shareable links.

5. **Restart your ASGI server** (Daphne / Uvicorn) to enable WebSocket endpoints.

---

## Testing

Run the test suite:

```bash
python manage.py test notes
```

The tests cover model constraints, API endpoints, permissions, filtering, pagination, CSV export, and webhook delivery.

---

## License

MIT License – see `LICENSE` file.

---

## Next Steps

- Integrate with a React frontend using the provided REST endpoints.
- Extend the knowledge graph with automatic paper linking via Semantic Scholar.
- Add more default templates for common ML workflows.

The NeuralOps Research Notebook is now ready to become your team’s single source of truth for all ML experiments.