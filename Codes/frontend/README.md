

# 🚀 NeuralOps Frontend

**A modern, cyber‑glass experiment management platform for attention compression research.**  
Real‑time monitoring, interactive visualizations, Overleaf‑style paper editing, hyperparameter sweeps, model registry, and GPU cluster dashboard.

![Design System](https://img.shields.io/badge/design-cyber--glass-blueviolet) ![React](https://img.shields.io/badge/react-18.2-blue) ![TypeScript](https://img.shields.io/badge/typescript-5.3-3178c6) ![Vite](https://img.shields.io/badge/vite-4.5-646cff)

---

## ✨ Features

### Phase 1 (Complete)
- **Dashboard** – KPI cards, recent experiments table, active jobs widget, system resource monitor  
- **Experiment Management** – Create runs (training/evaluation/compression), view details, stop/retry
- **Live Monitoring** – Real‑time loss charts, progress bar with ETA, log viewer via WebSocket
- **Interactive Launcher** – Model picker, task selector, attention mechanism picker, hyperparameter forms
- **Research Notebook** – Full‑featured notes with Markdown, reactions, sharing, version history, webhooks
- **Paper Editor** – Monaco‑based LaTeX editor with PDF preview (Overleaf‑style)
- **Terminal** – xterm.js terminal embedded in a floating window with PTY backend
- **Model Versioning** – Browse versions, publish to HuggingFace, compare models
- **Configuration History** – Versioned config snapshots with diff viewer
- **Custom Metrics Dashboard** – Plotly scalars, image gallery, tables
- **Admin Panel** – Job management, model version admin, bulk actions
- **Global Search** – Search across all entities (experiments, notes, models, jobs)
- **Activity Feed** – Audit log of platform actions
- **Authentication** – Token + cookie auth, login/register, profile, password change

### Phase 2 (Added)
- **Reproducibility Snapshots** – Auto‑capture git commit, environment, seed; downloadable `reproduce.py`
- **Run Comparison** – Side‑by‑side config diff, multi‑run metric overlay charts, parallel coordinates
- **Hyperparameter Sweeps** – Optuna‑powered wizard, trial dashboard, parameter importance plots, Hyperband pruning
- **Model Registry** – Lifecycle management (promote/archive), model cards, deployment stage badges
- **GPU Cluster Dashboard** – Utilization heatmap, per‑node GPU metrics, job queue depth
- **Enhanced WebSocket** – Reconnect with replay, batching, heartbeat, offline banner
- **Training Control Panel** – Start/stop with immediate feedback, live progress, connection status

---

## 🧰 Tech Stack

| Category | Technology |
|----------|------------|
| **Framework** | React 18 + TypeScript |
| **Build Tool** | Vite 4 |
| **Routing** | React Router 6 (lazy loading) |
| **State Management** | Zustand (auth, live store), React Query (server state) |
| **Styling** | Tailwind CSS 3 + custom glassmorphism components |
| **Charts** | Plotly.js (react-plotly.js), Chart.js |
| **Code Editor** | Monaco Editor (@monaco-editor/react) |
| **Terminal** | xterm.js + addons (fit, web-links, search) |
| **Markdown** | marked |
| **Icons** | Lucide React |
| **HTTP Client** | Axios with interceptors for token & 401 handling |
| **WebSocket** | Custom hook with auto‑reconnect |
| **Drag & Drop** | Floating windows (Zustand‑based) |

---

## 📁 Architecture & Folder Structure

```
src/
├── api/                   # Axios client & endpoint definitions
├── assets/                # Static assets
├── components/
│   ├── floating/          # Floating window manager (terminal, paper, notes, monitor)
│   ├── layout/            # MainLayout, ConnectionStatus
│   └── ui/                # Reusable UI primitives (Button, Card, Badge, etc.)
├── features/              # Feature‑based modules
│   ├── admin/             # Admin dashboard, jobs, model versions, users
│   ├── analytics/         # Advanced analytics (HP importance, metric correlation)
│   ├── auth/              # Login, register, store
│   ├── cluster/           # GPU cluster dashboard
│   ├── config/            # Dynamic configuration pages
│   ├── dashboard/         # Main dashboard (KPI, table, widgets)
│   ├── drift/             # Drift detection
│   ├── experiments/       # Experiment CRUD, monitoring, comparison, reproducibility
│   ├── fairness/          # Fairness evaluation
│   ├── hardware/          # System resource monitoring
│   ├── hpo/               # HPO status
│   ├── jobs/              # Job details, history, scheduling
│   ├── launcher/          # Experiment launcher (pickers, forms)
│   ├── modelRegistry/     # Model registry listing
│   ├── modelVersioning/   # Model versions, lineage, publication
│   ├── notes/             # Researcher notebook, rich text, sharing
│   ├── paper/             # Overleaf‑style editor, project management
│   ├── profile/           # User profile, password change
│   ├── prompts/           # Prompt registry
│   ├── robustness/        # Robustness evaluation
│   ├── scheduler/         # Cluster scheduler overview
│   ├── search/            # Global search
│   ├── sweeps/            # Hyperparameter sweeps (creator, dashboard)
│   ├── system/            # Health, version, system status
│   ├── templates/         # Experiment templates
│   └── terminal/          # Terminal panel (xterm)
├── hooks/                 # Global hooks (useWebSocket)
├── stores/                # Zustand stores (auth, experiment live, floating windows)
├── providers/             # AppProviders (QueryClient, Router)
├── types/                 # TypeScript interfaces
├── router.tsx             # Route definitions with lazy loading
├── App.tsx                # Root component
├── main.tsx               # Entry point
└── index.css              # Tailwind + custom glassmorphism styles
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js ≥ 18
- npm ≥ 9

### Installation
```bash
cd et-frontend
npm install
```

### Development
```bash
npm run dev
```
The app will start on `http://localhost:3000` and proxy API requests to `http://localhost:8000`.

### Production Build
```bash
npm run build
```
Outputs to `dist/`, ready to be served by Nginx or a CDN.

---

## ⚙️ Environment Variables

Create a `.env` file in the project root (or use `.env.development` / `.env.production`).

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:8000/api` |
| `VITE_WS_URL` | WebSocket base URL (ws/wss) | Automatically derived from window.location |

---

## 🎨 Design System

The UI follows a **cyber‑glass** aesthetic:
- **Colors:** Deep surface (`#0f172a`), neon accents (blue, purple, green), glassmorphism borders
- **Typography:** Inter (sans), JetBrains Mono (mono), Manrope (display)
- **Components:** Reusable `glass` card, `cyber‑grid` background, `shimmer` loading effects, `card-hover-effect`
- **Animations:** `fade-in`, `slide-up`, `scale-in`, `glow-pulse`

All reusable UI primitives are in `src/components/ui/` (Button, Card, Badge, Input, Select, etc.) and use `class-variance-authority` for variants.

---

## 🔌 Integration with Backend

The frontend communicates with the Django backend via:

- **REST API** – Axios client with `Token` authentication, auto‑logout on 401
- **WebSocket** – Custom `useWebSocket` hook for real‑time experiment updates, terminal I/O
- **Cookie Auth** – Token also stored in `authToken` cookie for middleware‑protected views

All endpoint URLs are centralised in `src/api/endpoints.ts`.

---

## 🔥 Phase 2 Enhancements

Phase 2 transforms the platform into a **production‑grade MLOps tool**:

- **Reproducibility Snapshots** – Immutable 4‑layer record (code, data, config, environment)  
- **Run Comparison** – Compare up to 20 runs with config diff table and metric overlay charts  
- **Hyperparameter Sweeps** – Optuna‑driven with TPE, SMAC3, CMA‑ES, Hyperband pruning  
- **Model Registry** – Staging → production → archive lifecycle with model cards  
- **GPU Cluster Dashboard** – DCGM + Prometheus integration, utilization heatmap, queue depths  
- **Enhanced WebSocket** – Replay on reconnect, batched metrics, heartbeat, offline detection  

All these features are accessible via the sidebar navigation and integrate seamlessly with Phase 1’s experiment workflow.

---

## 🧪 Testing

- Unit/component tests: `npm test` (Jest + React Testing Library)
- Integration tests: `src/test/integrationTest.ts` – uses Axios to verify backend health, system status, config endpoints
- E2E tests: Cypress (coming soon)

---

## 🤝 Contributing

We welcome contributions! Please follow the feature‑based folder structure and submit PRs with clear descriptions.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all existing pages still load correctly
5. Submit a pull request

---

## 📄 License

MIT License – see `LICENSE` file for details.

---

**Built with ❤️ for the neural research community.**  
For questions, open an issue or contact the maintainers.