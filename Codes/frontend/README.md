# 🚀 NeuralOps – Advanced ML Experiment Platform

**A complete, self‑hosted MLOps platform for attention compression research.**  
Track experiments, optimize hyperparameters, edit papers Overleaf‑style, ask an AI research assistant, and monitor GPU clusters – all from a beautiful dark‑glass interface.

![Design](https://img.shields.io/badge/design-cyber--glass-blueviolet)
![React](https://img.shields.io/badge/react-18.2-61dafb?logo=react)
![TypeScript](https://img.shields.io/badge/typescript-5.3-3178c6?logo=typescript)
![Django](https://img.shields.io/badge/django-4.2-092e20?logo=django)
![Vite](https://img.shields.io/badge/vite-4.5-646cff)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📖 Overview

NeuralOps is a **privacy‑first, on‑premise** alternative to cloud experiment trackers like W&B and MLflow.  
It integrates the entire ML lifecycle – from literature search to model deployment – into a single Django + React application.

**Why researchers choose NeuralOps:**

- **Zero per‑seat cost** – self‑host on your own GPU server or cluster
- **Real‑time collaboration** – shared notes, paper editor, integrated terminal
- **Full reproducibility** – automatic git/DVC/environment snapshots
- **AI‑powered** – conversational assistant helps search papers, configure experiments, and analyze training
- **Beautiful dark UI** – cyber‑glass design with live updates and interactive charts

---

## 🏗️ Architecture

```
┌──────────────────────┐          ┌────────────────────────────┐
│   React Frontend     │  REST    │      Django Backend         │
│ (Vite + TypeScript)  │◄────────►│  (DRF + Channels + Celery)  │
│  ┌────────────────┐  │ WebSocket│  ┌──────────┬─────────────┐ │
│  │ Floating       │◄──────────►│  │ REST API  │ WebSocket   │ │
│  │ Windows        │  │          │  │          │ Consumers   │ │
│  │ (terminal,     │  │          │  ├──────────┼─────────────┤ │
│  │  paper, notes) │  │          │  │ Celery    │ Optuna      │ │
│  └────────────────┘  │          │  │ Workers   │ Study       │ │
│                      │          │  └──────────┴─────────────┘ │
└──────────────────────┘          └─────────────────────────────┘
         │                                     │
         ▼                                     ▼
┌──────────────────┐              ┌──────────────────────────┐
│  PostgreSQL /    │              │   Kubernetes / Slurm     │
│  SQLite          │              │   (GPU scheduling)       │
└──────────────────┘              └──────────────────────────┘
```

---

## 🔥 Features

### Experiment Platform

| Feature                                                                                      | Status |
| -------------------------------------------------------------------------------------------- | ------ |
| **Dashboard** – KPI cards, recent experiments table, active jobs widget, system monitor      | ✅     |
| **Experiment Launcher** – Model, task, attention, method pickers with presets                | ✅     |
| **Live Monitoring** – Real‑time loss charts, progress bar with ETA, log viewer via WebSocket | ✅     |
| **Run Comparison** – Config diff table, multi‑run metric overlay, parallel coordinates       | ✅     |
| **Custom Metrics** – Plotly scalars, image gallery, tables                                   | ✅     |
| **Configuration History** – Versioned snapshots with diff viewer                             | ✅     |
| **Hyperparameter Sweeps** – Optuna‑powered with TPE, SMAC3, Hyperband pruning                | ✅     |

### AI Research Assistant

| Feature                  | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| **Conversational Chat**  | Ask questions, get experiment suggestions, and receive real‑time help       |
| **Paper Search**         | Search Semantic Scholar, arXiv, OpenAlex, and Consensus for relevant papers |
| **Citation Graph**       | Build interactive citation networks for discovered papers                   |
| **Paper Library**        | Save, tag, and annotate papers for future reference                         |
| **Training Analysis**    | Auto‑detect overfitting, plateau, divergence, and suggest fixes             |
| **Auto‑Note Generation** | Generate rich Markdown notes with loss charts at training checkpoints       |
| **Auto‑Stop**            | Automatically halt diverging experiments to save GPU hours                  |
| **Prompt Testing**       | Test prompt templates against multiple test cases                           |

### Collaboration & Knowledge Management

| Feature                 | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| **Research Notebook**   | Tiptap‑based rich text editor with Markdown, reactions, sharing, version history |
| **Paper Editor**        | Monaco‑based LaTeX editor with PDF compilation (Overleaf‑style)                  |
| **Integrated Terminal** | xterm.js floating terminal with PTY backend, auto‑reconnect, quick commands      |
| **Webhooks**            | Configure webhook subscriptions for note events                                  |
| **Activity Feed**       | Platform‑wide audit log of all actions                                           |

### Model Operations

| Feature                    | Description                                     |
| -------------------------- | ----------------------------------------------- |
| **Model Registry**         | Staging → production → archive lifecycle        |
| **Model Versioning**       | Automatic version creation on run completion    |
| **HuggingFace Publishing** | One‑click publish to HuggingFace Hub            |
| **Model Cards**            | Auto‑generated model cards from run metadata    |
| **Lineage Graph**          | Full data → code → model ancestry visualization |

### Cluster & Administration

| Feature                   | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| **GPU Cluster Dashboard** | Utilization heatmap, per‑node metrics, job queue depth |
| **Job Management**        | Bulk delete, stop all, priority scheduling             |
| **Admin Panel**           | Django Admin integration, system health monitoring     |
| **Statistics**            | Platform‑wide analytics with model and task usage      |

---

## 🧰 Technology Stack

### Backend (Python)

| Component                   | Technology                                     |
| --------------------------- | ---------------------------------------------- |
| Framework                   | Django 4.2 + Django REST Framework             |
| Async server                | Daphne (WebSocket support)                     |
| Real‑time                   | Django Channels                                |
| Task queue                  | Celery + Redis                                 |
| Hyperparameter optimization | Optuna 4.2                                     |
| LaTeX compilation           | pdflatex / xelatex / lualatex                  |
| Code versioning             | GitPython                                      |
| Database                    | PostgreSQL (production) / SQLite (development) |

### Frontend (TypeScript)

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| Framework        | React 18                                |
| State management | Zustand (client) + React Query (server) |
| Routing          | React Router 6 (lazy‑loaded routes)     |
| Styling          | Tailwind CSS 3 + custom glassmorphism   |
| Charts           | Plotly.js                               |
| Code editor      | Monaco Editor                           |
| Terminal         | xterm.js                                |
| Rich text        | Tiptap                                  |
| HTTP             | Axios with interceptors                 |
| WebSocket        | Custom hook with auto‑reconnect         |

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
│   ├── agents/                  # AI research assistant (tools, orchestrator, memory)
│   ├── web_dashboard/           # Django‑rendered dashboard (fallback)
│   ├── framework/               # Core ML framework (attention, compression, training)
│   ├── manage.py
│   └── requirements.txt
│
├── frontend/                    # React + TypeScript application
│   ├── src/
│   │   ├── api/                 # Axios client & endpoint definitions
│   │   ├── components/          # Shared UI (glass cards, buttons, floating windows)
│   │   ├── features/            # Feature modules (auth, dashboard, experiments, notes, ...)
│   │   ├── hooks/               # Custom hooks (useWebSocket)
│   │   ├── stores/              # Zustand stores
│   │   ├── router.tsx           # Route definitions (lazy loaded)
│   │   └── providers/           # AppProviders (QueryClient, Router)
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   └── tsconfig.json
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
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Development

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

1. Set `DEBUG=False`, `SECRET_KEY`, and `ALLOWED_HOSTS`
2. Use PostgreSQL and Redis
3. Collect static files: `python manage.py collectstatic`
4. Build frontend: `npm run build` → serve `dist/` via Nginx
5. Run Celery worker: `celery -A django_api worker -l info`
6. Run Celery beat: `celery -A django_api beat -l info`
7. Run Daphne behind Nginx with SSL

---

## 🔌 API Overview

All endpoints are prefixed with `/api/`. Key groups:

| Group          | Base Path                | Description                                                             |
| -------------- | ------------------------ | ----------------------------------------------------------------------- |
| Auth           | `/api/auth/`             | Login, register, logout, profile, password change                       |
| Experiments    | `/api/experiments/`      | CRUD, start/stop, compare, diff, reproducibility                        |
| Jobs           | `/api/jobs/`             | Job queue, scheduling, config history, custom metrics                   |
| Notes          | `/api/notes/`            | Full notebook CRUD, versioning, reactions, webhooks                     |
| Agent          | `/api/agents/`           | Chat sessions, paper search, library, citation graph, training analysis |
| Sweeps         | `/api/sweeps/`           | Create sweeps, list trials, importance plots                            |
| Model Registry | `/api/model-versioning/` | Model lifecycle, download, publish                                      |
| Paper          | `/api/paper/`            | LaTeX projects, compile, output PDF/log                                 |
| Terminal       | `/api/terminal/`         | Active sessions, kill session                                           |
| Cluster        | `/api/loadbalancer/`     | GPU allocator status, scheduler queue, cluster nodes                    |
| Config         | `/api/config/`           | Available models, tasks, attentions, compressors                        |
| Health         | `/api/health/`           | Backend health check                                                    |

---

## 🎨 Design System

- **Colors:** Deep surface (`#0f172a`), neon accents (blue, purple, green, orange)
- **Typography:** Inter (sans), JetBrains Mono (mono), Manrope (display)
- **Components:** `glass` cards, `cyber‑grid` background, `shimmer` loading, `card-hover-effect`
- **Animations:** fade‑in, slide‑up, scale‑in, glow‑pulse, float

---

## 🤖 AI Agent Tools

The conversational assistant has access to these tools:

| Tool                      | Description                                         |
| ------------------------- | --------------------------------------------------- |
| `search_papers`           | Search Semantic Scholar, arXiv, OpenAlex, Consensus |
| `get_paper_details`       | Get full paper metadata                             |
| `get_citation_graph`      | Build citation network                              |
| `save_paper`              | Save to library                                     |
| `search_library`          | Search saved papers                                 |
| `configure_experiment`    | Draft experiment configuration                      |
| `launch_experiment`       | Launch training (with confirmation)                 |
| `analyze_training`        | Detect overfitting, plateau, divergence             |
| `suggest_next_experiment` | Recommend next hyperparameters                      |
| `read_latex_project`      | Read LaTeX project files                            |
| `edit_latex_file`         | Edit LaTeX files                                    |
| `compile_latex`           | Compile to PDF                                      |
| `test_prompt_template`    | Test prompt templates                               |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License – see [LICENSE](./LICENSE) file for details.
