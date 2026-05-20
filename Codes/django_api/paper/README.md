Here's the commit message and the full README for the Paper App.

---

## Commit Message

```text
feat(paper): complete Overleaf-style LaTeX paper app with real-time collab, git sync, sharing, and templates

- Add PaperProject, PaperFile, PaperCompile, PaperTemplate, PaperShareLink,
  PaperComment, PaperVersion models with full indexing
- Implement full REST API: project CRUD, file CRUD with rename, clone,
  compile (pdflatex/xelatex/lualatex), output (PDF/log/preview), export ZIP,
  word count, collaboration, public share links, git push/pull, comments,
  version snapshots with restore, compile history, and HuggingFace upload
- Add WebSocket consumer for real-time collaborative editing (Channels)
- Add template system: users can browse public templates, create their own,
  and publish templates for others to use
- Add public template gallery with usage tracking
- Add LaTeX compiler engine with error/warning parsing
- Add admin panel registrations for all models
- Complete serializers with computed fields
- Full signals for cleanup
```

---

# 📝 Paper App – Overleaf-Style LaTeX Editor & Publisher

A complete, collaborative LaTeX paper management system built with Django REST Framework, Channels, and Celery.  
Write, compile, share, and publish your research papers directly from your browser.

---

## Table of Contents

- [📝 Paper App – Overleaf-Style LaTeX Editor \& Publisher](#-paper-app--overleaf-style-latex-editor--publisher)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Models](#models)
    - [Core Models](#core-models)
    - [Key Fields](#key-fields)
  - [API Reference](#api-reference)
    - [Projects](#projects)
    - [Files](#files)
    - [Compilation](#compilation)
    - [Templates](#templates)
    - [Collaboration](#collaboration)
    - [Share Links](#share-links)
    - [Comments](#comments)
    - [Version Snapshots](#version-snapshots)
    - [Git Integration](#git-integration)
    - [Export \& Utilities](#export--utilities)
    - [HuggingFace Upload](#huggingface-upload)
  - [Real-time Collaboration](#real-time-collaboration)
  - [Templates Gallery](#templates-gallery)
  - [Sharing \& Publishing](#sharing--publishing)
    - [Public Share Links](#public-share-links)
    - [HuggingFace Hub](#huggingface-hub)
  - [Git Integration](#git-integration-1)
  - [Version History](#version-history)
  - [Compilation Engine](#compilation-engine)
  - [Configuration](#configuration)
  - [Deployment](#deployment)
  - [Summary of All Endpoints](#summary-of-all-endpoints)

---

## Overview

The Paper App replicates the core features of Overleaf: you can create **projects** containing multiple `.tex`, `.bib`, and image files, compile them with `pdflatex`, `xelatex`, or `lualatex`, view the PDF inline, and download it. Multiple users can **collaborate in real-time** via WebSockets. Public **share links** let you send a read-only view to anyone, and one-click **HuggingFace publishing** gives you a permanent public URL.

---

## Features

- 📂 **Project Management** – create, rename, clone, delete projects
- 📄 **File Management** – create, edit, rename, upload binary files (images, PDFs)
- ⚡ **Real-time Collaboration** – multiple users edit the same project simultaneously (WebSockets)
- 🔧 **Multi-Compiler** – pdflatex, xelatex, lualatex with draft mode and error stopping
- 📊 **Compile Log Parsing** – automatic error/warning detection
- 📥 **PDF Preview & Download** – view compiled PDF inline, download output, view log
- 📈 **Word Count** – per-file and total word count
- 👥 **Collaborators** – add/remove collaborators with role-based permissions
- 🔗 **Public Share Links** – create password-protected, expiring share links (no login required)
- 📦 **Export** – download entire project as a ZIP archive
- 🕰️ **Version Snapshots** – save and restore project state at any point
- 💬 **Inline Comments** – add comments on specific files and line ranges
- 🔄 **Git Push/Pull** – sync project files with a remote Git repository
- 🏗️ **Templates Gallery** – browse public templates, use them to create projects, share your own
- 🚀 **HuggingFace Upload** – publish compiled PDF to HuggingFace Hub as a Dataset (free hosting)
- 🔍 **Full-Text Search** across projects (via Django ORM)
- 🛡️ **Fine-grained Permissions** – owner vs collaborators vs public

---

## Models

### Core Models

| Model | Description |
|-------|-------------|
| `PaperProject` | A LaTeX project with name, compiler settings, collaborators, git remote |
| `PaperFile` | A single file within a project (text or binary) |
| `PaperCompile` | A compilation record with status, output PDF, log, error/warning counts |
| `PaperTemplate` | A reusable template containing files as JSON |
| `PaperShareLink` | Public share link with token, password, expiration, view limits |
| `PaperComment` | Inline comment on a file and line range |
| `PaperVersion` | Full project snapshot at a point in time |

### Key Fields

**PaperProject**:
- `owner` – the creator, has full control
- `collaborators` – many-to-many users who can view and edit
- `root_file` – which `.tex` file is the main document (default: `main.tex`)
- `compiler` – `pdflatex`, `xelatex`, `lualatex`
- `draft_mode` – compile in draft mode (faster, no images)
- `stop_on_first_error` – halt compilation on first error
- `git_remote` – URL for Git push/pull
- `version` – auto-incremented on any file change

**PaperCompile**:
- `status` – `queued`, `running`, `completed`, `failed`, `timed_out`
- `pdf` – FileField containing the compiled PDF
- `log` – full compilation log text
- `error_count`, `warning_count` – parsed from log

**PaperShareLink**:
- `token` – unique URL-safe token
- `password` – optional access password
- `expires_at` – optional expiration date
- `max_views` – optional view limit

---

## API Reference

All endpoints use authentication (`IsAuthenticated`) unless marked as **Public**.

### Projects

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/paper/projects/` | List user's projects (paginated, searchable) |
| `POST` | `/api/paper/projects/` | Create a new project (optionally from template) |
| `GET` | `/api/paper/projects/{id}/` | Get project details (includes files, compiles, collaborators) |
| `PATCH` | `/api/paper/projects/{id}/` | Update project settings (owner only) |
| `DELETE` | `/api/paper/projects/{id}/` | Delete project (owner only) |
| `POST` | `/api/paper/projects/{id}/clone/` | Clone a project (own or accessible) |

### Files

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/paper/projects/{id}/files/` | List all files in a project |
| `POST` | `/api/paper/projects/{id}/files/` | Create or update a file (owner only) |
| `GET` | `/api/paper/projects/{id}/files/{filename}/` | Get file content |
| `PUT` | `/api/paper/projects/{id}/files/{filename}/` | Update file content (owner only) |
| `DELETE` | `/api/paper/projects/{id}/files/{filename}/` | Delete file (owner only) |
| `POST` | `/api/paper/projects/{id}/files/{filename}/rename/` | Rename a file (owner only) |

### Compilation

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/paper/projects/{id}/compile/` | Compile the project (body: `{compiler, draft_mode, stop_on_first_error}`) |
| `GET` | `/api/paper/projects/{id}/output/pdf/` | Download the latest compiled PDF |
| `GET` | `/api/paper/projects/{id}/output/log/` | View the compilation log |
| `GET` | `/api/paper/projects/{id}/preview/pdf/` | Stream PDF inline (for iframe embedding) |
| `GET` | `/api/paper/projects/{id}/compiles/` | Compile history (paginated) |

### Templates

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/paper/templates/` | List public and user's own templates |
| `POST` | `/api/paper/templates/` | Create a new template |
| `POST` | `/api/paper/templates/{id}/use/` | Create a new project from a template |

### Collaboration

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/paper/projects/{id}/collaborators/add/` | Add a collaborator by username (owner only) |
| `POST` | `/api/paper/projects/{id}/collaborators/remove/` | Remove a collaborator (owner only) |

### Share Links

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `POST` | `/api/paper/projects/{id}/share-links/` | Yes | Create a public share link |
| `GET` | `/api/paper/projects/{id}/share-links/list/` | Yes | List all share links for a project |
| `POST` | `/api/paper/projects/{id}/share-links/{link_id}/revoke/` | Yes | Revoke a share link |
| `GET` | `/api/paper/shared/{token}/` | **No** | View shared project details (public) |
| `GET` | `/api/paper/shared/{token}/pdf/` | **No** | Download PDF from shared link (public) |

### Comments

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/paper/projects/{id}/comments/` | List comments (filter by file, unresolved) |
| `POST` | `/api/paper/projects/{id}/comments/` | Add a comment |
| `PATCH` | `/api/paper/projects/{id}/comments/{comment_id}/` | Edit comment (author only) |
| `DELETE` | `/api/paper/projects/{id}/comments/{comment_id}/` | Delete comment (author only) |

### Version Snapshots

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/paper/projects/{id}/snapshots/` | List snapshots |
| `POST` | `/api/paper/projects/{id}/snapshots/` | Create a new snapshot (saves current state) |
| `GET` | `/api/paper/projects/{id}/snapshots/{version_number}/` | View a snapshot's file contents |
| `POST` | `/api/paper/projects/{id}/snapshots/{version_number}/restore/` | Restore project to a snapshot (owner only) |

### Git Integration

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/paper/projects/{id}/git/push/` | Push project files to configured git remote (owner only) |
| `POST` | `/api/paper/projects/{id}/git/pull/` | Pull project files from git remote (owner only) |

### Export & Utilities

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/paper/projects/{id}/export/` | Download entire project as a ZIP file |
| `GET` | `/api/paper/projects/{id}/wordcount/` | Get per-file and total word count |

### HuggingFace Upload

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/paper/projects/{id}/upload-hf/` | Upload compiled PDF to HuggingFace Hub (owner only) |

---

## Real-time Collaboration

**WebSocket endpoint:** `ws://<host>/ws/paper/{project_id}/?token=<auth_token>`

Users connected to the same project can edit files together in real-time:

- **`file_edit`** – broadcast content changes to all collaborators
- **`cursor`** – broadcast cursor positions (for showing other users' cursors)
- Changes are saved to the database automatically

---

## Templates Gallery

Users can **browse public templates** (e.g., "CVPR Paper", "IEEE Conference", "Preprint") and create projects from them with one click. They can also **publish their own templates** for others to use.

- `GET /api/paper/templates/` – returns a list of public and user's own templates
- `POST /api/paper/templates/` – create a new template (set `is_public: true` to share with the community)
- `POST /api/paper/templates/{id}/use/` – creates a new project pre-filled with the template's files

---

## Sharing & Publishing

### Public Share Links

Create a view-only link for anyone. Supports:
- Password protection (`password`)
- Expiration date (`expires_in_days`)
- Maximum number of views (`max_views`)

**Example:**
```json
{
  "expires_in_days": 7,
  "password": "review123",
  "max_views": 50
}
```
Returns: `{"url": "http://yourdomain.com/paper/shared/abc123/", "token": "abc123"}`

Anyone with the link can view the project's files and download the PDF (no login required).

### HuggingFace Hub

Compile your paper and publish it to HuggingFace as a Dataset. This gives you a permanent, free URL where anyone can view and download the PDF.

`POST /api/paper/projects/{id}/upload-hf/` → returns `{"url": "https://huggingface.co/datasets/username/paper-abc12345"}`

---

## Git Integration

Configure a Git remote URL in project settings (`git_remote` field). Then:

- **Push:** writes all project files to a temporary directory, initializes a Git repo, commits, and pushes to the remote.
- **Pull:** clones the remote into a temporary directory and imports all text files into the project.

This allows version control and sharing via GitHub, GitLab, or any Git host.

---

## Version History

Every time you change a file, the project version increments. Additionally, you can manually create **snapshots** at any point (e.g., before a major rewrite). Snapshots store the full content of all files.

- `POST /api/paper/projects/{id}/snapshots/` – create a snapshot
- `GET /api/paper/projects/{id}/snapshots/{version_number}/` – view snapshot contents
- `POST /api/paper/projects/{id}/snapshots/{version_number}/restore/` – roll back to a previous snapshot

---

## Compilation Engine

The `LatexCompiler` class handles:
- Writing all project files to a temporary directory
- Running `pdflatex`/`xelatex`/`lualatex` with appropriate flags
- Parsing the log file for errors, warnings, and bad boxes
- Timeout enforcement (configurable, default 60 seconds)
- Collecting the output PDF and log

---

## Configuration

Add to `settings.py`:

```python
# LaTeX compilers (must be installed on the system)
LATEX_COMPILERS = {
    'pdflatex': '/usr/bin/pdflatex',
    'xelatex': '/usr/bin/xelatex',
    'lualatex': '/usr/bin/lualatex',
}

# HuggingFace Hub (optional)
HUGGINGFACE_TOKEN = "hf_..."
HUGGINGFACE_USERNAME = "your-username"

# Public base URL (used for share links)
PUBLIC_BASE_URL = "https://your-domain.com"

# Channels (for real-time collaboration)
ASGI_APPLICATION = "your_project.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [("redis", 6379)]},
    },
}
```

**Install system dependencies:**

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex
```

**Install Python dependencies:**

```bash
pip install channels channels-redis huggingface-hub
```

---

## Deployment

1. **Run migrations:**
   ```bash
   python manage.py makemigrations paper
   python manage.py migrate
   ```

2. **Add the WebSocket routing** to your main ASGI application.

3. **Set up Redis** for Channels (required for real-time collaboration).

4. **Configure `ALLOWED_HOSTS`** and `PUBLIC_BASE_URL` for shareable links.

5. **Restart your ASGI server** (Daphne / Uvicorn).

---

## Summary of All Endpoints

| Category | Count | Key Features |
|----------|-------|-------------|
| Projects | 5 | CRUD, clone |
| Files | 6 | CRUD, rename, binary upload |
| Compilation | 5 | Compile, PDF, log, preview, history |
| Templates | 3 | List, create, use |
| Collaboration | 2 | Add/remove collaborators |
| Share Links | 5 | Create, list, revoke, public view, public PDF |
| Comments | 4 | CRUD with filtering |
| Snapshots | 4 | Create, list, view, restore |
| Git | 2 | Push, pull |
| Export/Utils | 2 | ZIP download, word count |
| HuggingFace | 1 | Publish compiled PDF |

