# 🧬 NeuralOps – Advanced ML Experiment Platform

**Transform your attention‑compression research with a complete, self‑hosted MLOps platform.**  
Track experiments, optimize hyperparameters, edit papers in Overleaf‑style, manage model registries, and monitor GPU clusters – all from a single, beautiful cyber‑glass interface.

[![Django](https://img.shields.io/badge/django-4.2-092e20?logo=django)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/react-18.2-61dafb?logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.3-3178c6?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/tailwind-3.4-38bdf8?logo=tailwindcss)](https://tailwindcss.com/)
[![Channels](https://img.shields.io/badge/channels-4.1-blueviolet)](https://channels.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/optuna-4.2-3d5a80)](https://optuna.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## 📖 Overview

NeuralOps is a **privacy‑first, on‑premise** alternative to cloud‑based experiment trackers.  
It integrates the entire ML lifecycle – from hyperparameter sweeps to model deployment – into a single Django + React application.

**Why researchers love it:**
- **Zero per‑seat cost** – self‑host on your GPU server or cluster.
- **Real‑time collaboration** – shared notes, paper editor, and terminal.
- **Reproducibility** – automatic git/DVC/environment snapshots.
- **Beautiful UI** – cyber‑glass design with live updates and interactive charts.

---

## 🏗️ High‑Level Architecture

```
┌─────────────────────┐          ┌────────────────────────────┐
│   React Frontend    │  REST    │      Django Backend        │
│ (Vite + TypeScript) │◄────────►│  (DRF + Channels + Celery) │
│  ┌───────────────┐  │ WebSocket│  ┌──────────┬────────────┐ │
│  │ Floating      │◄──────────►│  │ REST API │ WebSocket  │ │
│  │ Windows       │  │          │  │          │ Consumers  │ │
│  │ (terminal,    │  │          │  ├──────────┼────────────┤ │
│  │  paper, notes)│  │          │  │ Celery   │ Optuna     │ │
│  └───────────────┘  │          │  │ Workers  │ Study      │ │
│                     │          │  └──────────┴────────────┘ │
└─────────────────────┘          └────────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
┌─────────────────┐              ┌────────────────────────────┐
│  PostgreSQL /   │              │     Kubernetes / Slurm     │
│  SQLite         │              │     (GPU scheduling)       │
└─────────────────┘              └────────────────────────────┘
```

---

## 🔥 Features

### Phase 1 – Core Experiment Platform
| Feature | Status |
|---------|--------|
| **Experiment Dashboard** – KPI cards, recent runs table, active jobs widget | ✅ |
| **Experiment Launcher** – Model, task, attention, method pickers with presets | ✅ |
| **Live Monitoring** – Real‑time loss charts, progress bar with ETA, log viewer via WebSocket | ✅ |
| **Research Notebook** – Markdown notes with reactions, sharing, version history, webhooks | ✅ |
| **Paper Editor** – Monaco‑based LaTeX editor with PDF compilation (Overleaf‑style) | ✅ |
| **Integrated Terminal** – xterm.js floating terminal with PTY backend | ✅ |
| **Model Versioning** – Version history, publish to HuggingFace, download artifacts | ✅ |
| **Custom Metrics** – Plotly scalars, image gallery, tables | ✅ |
| **Configuration History** – Versioned snapshots with diff viewer | ✅ |
| **Admin Panel** – Jobs management, model versions, bulk actions | ✅ |
| **Global Search** – Search across experiments, notes, models, jobs | ✅ |
| **Authentication** – Token + cookie auth, profile, password change | ✅ |

### Phase 2 – Advanced MLOps Features
| Feature | Status |
|---------|--------|
| **Reproducibility Snapshots** – Git commit, DVC hash, environment capture, auto‑generated `reproduce.py` | ✅ |
| **Run Comparison** – Config diff table, multi‑run metric overlay, parallel coordinates | ✅ |
| **Hyperparameter Sweeps** – Optuna‑powered with TPE, SMAC3, CMA‑ES, Hyperband pruning, trial dashboard | ✅ |
| **Model Registry** – Staging → production → archive lifecycle, model cards, promote/archive actions | ✅ |
| **GPU Cluster Dashboard** – Utilization heatmap, per‑node metrics, job queue depth | ✅ |
| **Enhanced WebSocket** – GZIP compression, batched metrics, replay on reconnect, heartbeat | ✅ |
| **Lineage Graph** – Full data‑to‑model ancestry visualization | ✅ |

---

## 🧰 Technology Stack

### Backend (Python)
| Component | Technology |
|-----------|------------|
| Framework | Django 4.2 + Django REST Framework |
| Async server | Daphne (for WebSockets) |
| WebSockets | Django Channels |
| Task queue | Celery + Redis |
| Hyperparameter optimization | Optuna 4.2 |
| LaTeX compilation | pdflatex / xelatex / lualatex |
| Code versioning | GitPython (auto‑capture git info) |
| Database | PostgreSQL (production) / SQLite (development) |

### Frontend (TypeScript)
| Component | Technology |
|-----------|------------|
| Framework | React 18 |
| State management | Zustand (client) + React Query (server) |
| Routing | React Router 6 (lazy routes) |
| Styling | Tailwind CSS 3 + custom glass‑morphism |
| Charts | Plotly.js (react-plotly.js) |
| Code editor | Monaco Editor |
| Terminal | xterm.js |
| Rich text | Tiptap |
| HTTP | Axios with interceptors |

---

## 📁 Project Structure

```
neuralops/
├── backend/                     # Django project
│   ├── api/                     # Main API (auth, config, system, mega launcher)
│   ├── experiment/              # Experiment models, runners, comparisons
│   ├── jobscheduler/            # Job queue, priority, scheduling, custom metrics
│   ├── notes/                   # Research notebook (CRUD, versioning, sharing)
│   ├── model_versioning/        # Model registry, lifecycle, publication
│   ├── sweeps/                  # Optuna hyperparameter sweep engine
│   ├── terminal/                # Interactive terminal (Channels consumer)
│   ├── paper/                   # LaTeX project management & compilation
│   ├── web_dashboard/           # Django‑rendered dashboard (fallback)
│   ├── framework/               # Core ML framework (attention, compression, training)
│   ├── manage.py                # Django management
│   └── requirements.txt         # Python dependencies
│
├── frontend/                    # React + TypeScript application
│   ├── src/
│   │   ├── api/                 # Axios client & endpoint definitions
│   │   ├── components/          # Shared UI (glass cards, buttons, floating windows)
│   │   ├── features/            # Feature modules (auth, dashboard, experiments, notes, ...)
│   │   ├── hooks/               # Custom hooks (useWebSocket)
│   │   ├── stores/              # Zustand stores
│   │   ├── router.tsx           # Route definitions (lazy loaded)
│   │   └── App.tsx              # Root component
│   ├── package.json
│   └── tailwind.config.js
│
├── docker-compose.yml           # Production deployment (optional)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- **Python** ≥ 3.10
- **Node.js** ≥ 18
- **Redis** (for Celery & Channels in production)
- **pdflatex** (optional, for paper compilation)
- **NVIDIA GPU + CUDA** (optional, for training)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running in Development

Start the backend (with WebSocket support):
```bash
cd backend
daphne django_api.asgi:application
```

In a separate terminal, start the frontend:
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** and log in with your superuser credentials.

### Production Deployment

1. Set `DEBUG=False`, `SECRET_KEY`, and `ALLOWED_HOSTS` in `backend/django_api/settings.py`
2. Use PostgreSQL and Redis
3. Collect static files: `python manage.py collectstatic`
4. Build frontend: `npm run build` → serve `dist/` via Nginx
5. Run Celery workers: `celery -A django_api worker -l info`
6. Run Daphne behind Nginx with SSL

---

## 🔌 API Overview

All endpoints are prefixed with `/api/`. Key groups:

| Group | Base Path | Description |
|-------|-----------|-------------|
| Auth | `/api/auth/` | Login, register, logout, profile, password change |
| Experiments | `/api/experiments/` | CRUD, start/stop, compare, diff, reproducibility |
| Jobs | `/api/jobs/` | Job queue, scheduling, config history, custom metrics |
| Notes | `/api/notes/` | Full notebook CRUD, versioning, reactions, webhooks |
| Sweeps | `/api/sweeps/` | Create sweeps, list trials, importance plots |
| Model Registry | `/api/model-versioning/` | Model lifecycle, download, publish |
| Paper | `/api/paper/` | LaTeX projects, compile, output PDF/log |
| Terminal | `/api/terminal/` | Active sessions, kill session |
| Cluster | `/api/cluster/` | GPU nodes, utilization, queue depth |
| Config | `/api/config/` | Available models, tasks, attentions, compressors |
| Load Balancer | `/api/loadbalancer/` | GPU allocator status, scheduler queue |
| Health | `/api/health/` | Backend health check |

---

## 🤝 Contributing

We welcome contributions! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
