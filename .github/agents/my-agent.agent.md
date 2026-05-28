# 🧠 NeuralOps — Complete Project Documentation

**A Decentralized GPU Marketplace & ML Research Platform with Social Attention Visualization**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Backend: Django Apps](#3-backend-django-apps)
4. [Frontend: React + Electron](#4-frontend-react--electron)
5. [Provider Agent](#5-provider-agent)
6. [API Reference](#6-api-reference)
7. [Setup & Installation](#7-setup--installation)
8. [Development Workflow](#8-development-workflow)
9. [Deployment](#9-deployment)
10. [Contributing](#10-contributing)

---

## 1. Project Overview

NeuralOps is a **comprehensive ML research platform** that combines:

- **Decentralized GPU Marketplace** — Providers share idle GPU resources; consumers rent compute by the hour. Includes order books, matching engine, escrow, verification, benchmarks, futures trading, and reputation systems.
- **Experiment Management** — Launch, monitor, and analyze training, evaluation, and compression experiments with real‑time WebSocket dashboards.
- **Research Notebook** — Collaborative Markdown/LaTeX notes with version history, templates, and HuggingFace publishing.
- **Paper Editor** — Overleaf‑style LaTeX editor with real‑time collaboration, pdflatex/xelatex/lualatex compilation, and Git sync.
- **Autonomous Research Agent** — Multi‑agent LangGraph system for paper search, hypothesis generation, and experiment suggestions.
- **Community Q&A** — Stack Overflow‑style forum with reputation, badges, bounties, and Discourse‑style trust levels.
- **Leaderboards & Benchmarks** — PapersWithCode‑style public benchmarks with Kaggle‑style competitions.
- **Peer Review System** — OpenReview‑style structured peer review with Kleros‑inspired jury arbitration.
- **Model Versioning** — Automatic artifact compression, diffing, cold storage, and HuggingFace publishing.
- **Organizations & Teams** — Multi‑tenant workspaces with RBAC, SSO, billing plans, and audit logs.
- **Attention Visualization Lab** — 6 interactive views (heatmap, model grid, 3D, Sankey, Q/K joint, token animation) with live math, ablation tools, and social features.
- **Provider Desktop App** — Electron + React GUI with system tray, real‑time earnings, GPU monitoring, and job control.

---

## 2. High‑Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEURALOPS PLATFORM                                │
│                                                                          │
│  ┌──────────────────────┐    ┌────────────────────────────────────────┐ │
│  │   React SPA (Vite)   │    │        Electron Desktop App            │ │
│  │   (Consumer Web UI)  │    │    (Provider GPU Agent GUI)            │ │
│  │                      │    │                                        │ │
│  │  • Dashboard         │    │  • Live GPU Monitor                   │ │
│  │  • Marketplace       │    │  • Earnings Dashboard                 │ │
│  │  • Experiments       │    │  • Job Control                        │ │
│  │  • Attention Lab     │    │  • System Tray                        │ │
│  │  • Notes / Paper     │    │  • Real‑time Logs                     │ │
│  │  • Community         │    │                                        │ │
│  │  • Leaderboards      │    │                                        │ │
│  │  • Profile           │    │                                        │ │
│  └──────────┬───────────┘    └──────────────────┬─────────────────────┘ │
│             │                                   │                       │
│             │         REST API + WebSocket       │                       │
│             └───────────────┬───────────────────┘                       │
│                             │                                           │
│  ┌──────────────────────────┴──────────────────────────────────────┐   │
│  │                    Django Backend (ASGI)                          │   │
│  │                                                                   │   │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │   │
│  │  │  Auth   │ │  Users   │ │  Social  │ │  Notes   │ │  Paper  │ │   │
│  │  │ (DRF)   │ │(Profiles)│ │ (Allauth)│ │(Markdown)│ │(LaTeX)  │ │   │
│  │  └─────────┘ └──────────┘ └──────────┘ └──────────┘ └─────────┘ │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │              GPU Marketplace Suite                           │ │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐  │ │   │
│  │  │  │Marketplace│ │ Escrow   │ │ Validator │ │ Benchmark    │  │ │   │
│  │  │  │(OrderBook)│ │(Payments)│ │(Checker)  │ │(MLPerf)      │  │ │   │
│  │  │  └──────────┘ └──────────┘ └───────────┘ └──────────────┘  │ │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐  │ │   │
│  │  │  │Federation│ │ Finance  │ │Reputation │ │ Organizations│  │ │   │
│  │  │  │(DistTrain)│ │(Futures) │ │(Staking)  │ │(Teams/RBAC)  │  │ │   │
│  │  │  └──────────┘ └──────────┘ └───────────┘ └──────────────┘  │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                   │   │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────────┐  │   │
│  │  │Community │ │Leaderboard│ │ Reviews  │ │Model Versioning   │  │   │
│  │  │(Q&A)     │ │(Benchmarks)│ │(Peer Rev)│ │(Artifact Mgmt)    │  │   │
│  │  └──────────┘ └───────────┘ └──────────┘ └───────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                             │                                           │
│  ┌──────────────────────────┴──────────────────────────────────────┐   │
│  │                    Python Provider Agent                          │   │
│  │  • GPU Detection (pynvml)    • Docker Container Runtime          │   │
│  │  • Network Speed Test        • WebSocket Heartbeat               │   │
│  │  • DLPerf Benchmark Runner   • Auto‑Update Mechanism             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Backend: Django Apps

### 3.1 `api` — Core API & Authentication

**Purpose:** REST API entry point, authentication, system endpoints.

**Key Files:**
- `views.py` — Registration, login, logout, user info, profile update, password change, token refresh
- `authentication.py` — `CookieTokenAuthentication` (DRF Token + cookie fallback)
- `consumers.py` — `JobConsumer`, `TerminalConsumer` (WebSocket)
- `routing.py` — WebSocket URL routing
- `urls.py` — All top‑level API routes aggregated

**Authentication Flow:**
1. User registers/logs in → receives DRF `Token`
2. Token stored in `localStorage` + `authToken` cookie
3. All API calls send `Authorization: Token <key>`
4. WebSocket connections authenticate via query string `?token=<key>`

---

### 3.2 `profiles` — User Profiles

**Purpose:** Extended user profiles with avatars, bios, skills, research interests, social links, and GitHub/HuggingFace OAuth.

**Models:**
- `Profile` — 1‑to‑1 with `User`, contains avatar, bio, affiliation, skills, social links, ORCID

**Key Endpoints:**
- `GET/PATCH /api/profiles/me/` — View/update own profile
- `POST /api/profiles/me/avatar/` — Upload avatar
- `GET /api/profiles/search/?q=...` — Search users
- `GET /api/profiles/user/{id}/` — View any user's profile
- `POST /api/profiles/social-callback/` — Exchange OAuth code for DRF token

**Social Login Providers:**
- GitHub (via `django-allauth`)
- HuggingFace (custom provider at `profiles/providers/huggingface/`)

---

### 3.3 `messaging` — Private Chat

**Purpose:** Real‑time private messaging between users.

**Models:** `ChatRoom`, `Message`

**Key Endpoints:**
- `GET /api/messaging/chats/` — List chat rooms
- `POST /api/messaging/chats/start/` — Start or get existing chat
- `POST /api/messaging/chats/{id}/send/` — Send message (REST fallback)
- `GET /api/messaging/chats/{id}/` — Get messages (marks read)

**WebSocket:** `ws/chat/{room_id}/?token=<auth_token>` — Real‑time messaging with typing indicators and presence.

---

### 3.4 `experiment` — Experiment Management

**Purpose:** Training, evaluation, and compression experiment lifecycle.

**Models:** `Experiment`, `TrainingRun`, `TrainingSnapshot`

**Runners:** `run_training()`, `run_evaluation()`, `run_compression()` — execute jobs via Celery, stream progress via WebSocket.

**Key Endpoints:**
- `GET/POST /api/experiments/` — List/create experiments
- `POST /api/runs/start-train/` — Start training job
- `GET /api/runs/{id}/` — Run detail
- `POST /api/runs/{id}/stop/` — Stop running job
- `GET /api/runs/{id}/results/` — Results & artifacts

---

### 3.5 `jobscheduler` — GPU Job Scheduler

**Purpose:** GPU‑aware job scheduler with backfill, fair‑share, and aging.

**Models:** `Job`

**Key Features:**
- GPU allocation with `GPUScheduler v2`
- Backfill scheduling for short jobs
- Fair‑share across users
- Preemption with configurable timeouts
- Priority queue with aging

---

### 3.6 `gpu_marketplace` — GPU Exchange

**Purpose:** Two‑sided marketplace for GPU compute — providers list GPUs, consumers place orders.

**Models:**
- `GPUModelSpec` — Reference GPU specifications
- `GPUListing` — A GPU resource listed for rent
- `ComputeOrder` — Buy/sell order with market/limit/auction/future modes
- `ComputeJob` — A running/completed compute instance
- `Bid` — Auction bids
- `JobEvent` — Audit trail for job lifecycle
- `PricingRule` — Dynamic pricing configuration

**Matching Engine:** Price‑Time Priority (PTP) algorithm in `matching_engine.py` — matches buy orders with sell orders every 30 seconds via Celery Beat.

**Key Endpoints:**
- `GET /api/gpu/listings/` — Search GPU listings (filter by model, VRAM, price, reliability)
- `POST /api/gpu/listings/` — Provider creates listing
- `POST /api/gpu/orders/` — Place buy/sell order
- `GET /api/gpu/jobs/` — List my jobs
- `POST /api/gpu/jobs/{id}/complete/` — Provider marks job complete

---

### 3.7 `gpu_escrow` — Secure Payments

**Purpose:** Escrow system for holding and releasing payments.

**Models:** `EscrowAccount`, `EscrowTransaction`

**Flow:**
1. Consumer deposits funds → `balance_usd` increases
2. Job matched → funds move from `balance_usd` to `locked_balance_usd`
3. Job completed → funds released from consumer's locked balance to provider's available balance
4. Job failed → funds refunded to consumer's available balance

**Key Endpoints:**
- `GET /api/escrow/balance/` — View balance
- `POST /api/escrow/deposit/` — Add funds
- `POST /api/escrow/withdraw/` — Withdraw available funds
- `GET /api/escrow/transactions/` — Transaction history

---

### 3.8 `gpu_validator` — Trust Verification

**Purpose:** Aethir‑style checker nodes that verify compute integrity.

**Models:** `ValidatorNode`, `VerificationTask`, `Verification`, `ProofOfCompute`

**Flow:**
1. Job completes → `VerificationTask` created
2. Multiple validators submit results
3. Consensus reached (supermajority 2/3)
4. Validators rewarded/punished based on alignment with consensus

**Key Endpoints:**
- `POST /api/validator/validators/register/` — Register as validator (stake required)
- `POST /api/validator/tasks/{id}/submit/` — Submit verification result
- `POST /api/validator/tasks/{id}/finalize/` — Trigger consensus check

---

### 3.9 `gpu_benchmark` — MLPerf‑Style Performance Testing

**Purpose:** Standardized GPU benchmarking with composite DLPerf scoring.

**Models:** `BenchmarkSuite`, `BenchmarkTest`, `BenchmarkRun`, `BenchmarkResult`, `ListingBenchmarkScore`

**Benchmark Suites:** DLPerf Standard v1.0 (ResNet‑50, BERT‑Large, GPT‑2), DLPerf Inference v1.0, GPU Stress Test v1.0.

**Key Endpoints:**
- `GET /api/benchmarks/suites/` — List benchmark suites
- `POST /api/benchmarks/listings/{id}/run/` — Start benchmark
- `GET /api/benchmarks/leaderboard/` — GPU performance leaderboard

---

### 3.10 `gpu_federation` — Distributed Training

**Purpose:** Petals/Hivemind‑style distributed training across multiple GPU nodes.

**Models:** `FederationCluster`, `FederationNode`, `FederationTask`, `FederationTaskShard`, `FederationResult`

**Sharding Strategies:** Layer‑wise, data‑parallel, tensor‑parallel, hybrid.

**Key Endpoints:**
- `POST /api/federation/clusters/` — Create cluster
- `POST /api/federation/clusters/{id}/nodes/` — Add GPU to cluster
- `POST /api/federation/tasks/{id}/launch/` — Launch distributed training

---

### 3.11 `gpu_finance` — Financial Markets

**Purpose:** CME‑style compute futures, billing, invoicing, subscriptions.

**Models:** `ComputeIndex`, `FuturesContract`, `FuturesPosition`, `FuturesOrder`, `Invoice`, `InvoiceLineItem`, `Payout`, `PaymentMethod`, `BillingPlan`, `Subscription`, `UsageRecord`

**Key Endpoints:**
- `GET /api/finance/indices/` — Compute indices
- `GET /api/finance/futures/` — Active futures contracts
- `GET /api/finance/invoices/` — My invoices
- `GET /api/finance/usage/` — My usage records

---

### 3.12 `gpu_reputation` — Trust & Dispute Resolution

**Purpose:** EigenLayer‑style slashing, Kleros‑inspired jury arbitration, OpenRank EigenTrust.

**Models:** `ReputationScore`, `Stake`, `SlashingCondition`, `SlashingEvent`, `Dispute`, `JuryMember`, `JuryVote`, `ReputationBadge`, `UserReputationBadge`, `TrustEdge`, `EigenTrustResult`

**Key Endpoints:**
- `GET /api/reputation/scores/{username}/` — User's reputation
- `POST /api/reputation/stakes/stake/` — Stake collateral
- `POST /api/reputation/disputes/raise/{job_id}/` — Raise dispute
- `POST /api/reputation/disputes/{id}/vote/` — Cast juror vote

---

### 3.13 `organizations` — Teams & RBAC

**Purpose:** Multi‑tenant workspaces with role‑based access control.

**Models:** `Organization`, `Role`, `Membership`, `Team`, `TeamMembership`, `Invitation`, `ResourceGroup`, `AuditLog`, `Plan`, `Subscription`

**Built‑in Roles:** Owner, Admin, Member, Guest — with granular permissions.

**Key Endpoints:**
- `POST /api/orgs/orgs/` — Create organization
- `POST /api/orgs/orgs/{id}/invite/` — Invite member
- `GET /api/orgs/orgs/{id}/audit-log/` — Compliance audit trail

---

### 3.14 `community` — Research Q&A

**Purpose:** Stack Overflow‑style forum with reputation, badges, bounties, and Discourse‑style trust levels.

**Models:** `Tag`, `Question`, `Answer`, `Comment`, `Vote`, `Bookmark`, `Bounty`, `ReputationEvent`, `TrustLevel`, `Badge`, `UserBadge`, `Privilege`, `ActivityLog`, `DigestSubscription`

**Reputation System:**
- Question upvoted: +5, Answer upvoted: +10, Answer accepted: +15
- Downvoted answer: −2
- Bounty offered: −50 to −500

**Key Endpoints:**
- `GET/POST /api/community/questions/` — List/ask questions
- `POST /api/community/questions/{id}/answer/` — Post answer
- `POST /api/community/questions/{id}/vote/` — Vote
- `POST /api/community/questions/{id}/bounty/` — Start bounty

---

### 3.15 `leaderboards` — Benchmarks & Competitions

**Purpose:** PapersWithCode‑style public benchmarks with Kaggle‑style competitions.

**Models:** `BenchmarkTask`, `BenchmarkDataset`, `Metric`, `Benchmark`, `Submission`, `Score`, `Vote`, `Badge`, `UserBadge`, `BenchmarkFollow`

**Key Endpoints:**
- `GET /api/leaderboards/benchmarks/` — List benchmarks
- `POST /api/leaderboards/benchmarks/{id}/submissions/` — Submit results
- `GET /api/leaderboards/benchmarks/{id}/leaderboard/` — Rankings

---

### 3.16 `reviews` — Peer Review

**Purpose:** OpenReview‑style structured peer review with Kleros jury.

**Models:** `ReviewRequest`, `Review`, `ReviewComment`, `ReviewTemplate`, `ReviewBadge`, `ReviewerBadge`, `ReviewVote`

**Key Endpoints:**
- `POST /api/reviews/requests/` — Submit for review
- `POST /api/reviews/requests/{id}/review/` — Write review
- `GET /api/reviews/requests/{id}/summary/` — Aggregated scores

---

### 3.17 `notes` — Research Notebook

**Purpose:** Collaborative Markdown/LaTeX notebook with version history, templates, auto‑report generation, and HuggingFace publishing.

**Models:** `ResearcherNote`, `NoteVersion`, `NoteImage`, `NoteMention`, `NoteComment`, `NoteTemplate`, `ShareableLink`, `Artifact`, `Protocol`, `NoteLink`

**Key Features:**
- Auto‑generate training reports from `TrainingRun`
- Full‑text search (PostgreSQL `SearchVector`)
- Public share links with password, expiration, max views
- Upload to HuggingFace as Dataset or static Space

**Key Endpoints:** 50+ endpoints under `/api/notes/v1/`

---

### 3.18 `paper` — LaTeX Paper Editor

**Purpose:** Overleaf‑style collaborative LaTeX editor with real‑time collaboration, compilation, and Git sync.

**Models:** `PaperProject`, `PaperFile`, `PaperCompile`, `PaperTemplate`, `PaperShareLink`, `PaperComment`, `PaperVersion`

**Compilers:** pdflatex, xelatex, lualatex

**Key Endpoints:**
- `POST /api/paper/projects/{id}/compile/` — Compile project
- `GET /api/paper/projects/{id}/output/pdf/` — Download PDF
- `POST /api/paper/projects/{id}/github/push/` — Push to GitHub
- `POST /api/paper/projects/{id}/upload-hf/` — Publish to HuggingFace

---

### 3.19 `agents` — Autonomous Research Agent

**Purpose:** Multi‑agent LangGraph system for paper search, hypothesis generation, and experiment suggestions.

**Models:** `ChatSession`, `ChatMessage`, `ResearchPaper`, `CitationEdge`, `TrainingAnalysis`

**Tools:** 15+ tools including `search_papers`, `save_paper`, `analyze_training`, `suggest_next_experiment`, `configure_experiment`, `launch_experiment`

**Key Endpoints:**
- `POST /api/chats/{id}/send/` — Send message to agent
- `POST /api/papers/search/` — Search Semantic Scholar/arXiv/OpenAlex
- `POST /api/training/analyze/` — Analyze training run

---

### 3.20 `portfolios` — Researcher Showcase

**Purpose:** Public portfolio pages for researchers with models, papers, badges, and activity graphs.

**Models:** `Portfolio`, `ShowcaseItem`, `Endorsement`, `ActivityEvent`, `WidgetEmbed`

**Key Endpoints:**
- `GET /api/portfolios/user/{username}/` — Public portfolio
- `POST /api/portfolios/me/items/` — Add showcase item
- `GET /api/portfolios/user/{username}/activity/` — GitHub‑style contribution chart
- `GET /api/portfolios/user/{username}/embed/` — Embeddable widget

---

### 3.21 `web_dashboard` — Consumer Dashboard API

**Purpose:** Dashboard widgets, marketplace consumer endpoints, launch configuration.

**Models:** `DashboardWidget` (15+ widget types)

**Marketplace Consumer Endpoints:**
- `GET /api/marketplace/listings/` — Browse GPU listings
- `POST /api/marketplace/orders/` — Place buy order
- `GET /api/marketplace/jobs/` — My compute jobs
- `GET /api/marketplace/balance/` — Escrow balance
- `GET /api/marketplace/leaderboard/` — GPU leaderboard

---

### 3.22 Supporting Apps

| App | Purpose |
|-----|---------|
| `model_versioning` | Artifact compression, diffing, cold storage, HF publishing |
| `sweeps` | Hyperparameter optimization with Optuna |
| `terminal` | Web‑based terminal for job interaction |

---

## 4. Frontend: React + Electron

### 4.1 Web Dashboard (`frontend/`)

**Stack:** React 18, TypeScript, Vite, Tailwind CSS, TanStack Query, Recharts, Plotly, React Router v6

**Key Features:**
- **Dashboard** — Customizable widget grid with 15+ widget types
- **Experiments** — List, detail, monitor (real‑time WebSocket), compare, diff
- **Launcher** — Model/task/attention/method configuration wizard
- **Notes** — Full CRUD, version history, templates, PDF export, bulk operations
- **Paper** — Overleaf‑style editor with file tree, real‑time collab, PDF preview
- **Marketplace** — GPU listings, orders, jobs, billing, leaderboard
- **Attention Lab** — 6 interactive views, live math panel, ablation tools, social features
- **Profile** — Avatar, bio, skills, social links
- **Admin** — Jobs, users, model versions management
- **System** — Health, GPU info, version, cluster overview

**Routing:** 80+ routes managed by `react-router-dom` with lazy loading.

---

### 4.2 Electron Desktop App (`neuralops-gui/`)

**Purpose:** Desktop application for GPU providers to monitor earnings, manage jobs, and control their GPU.

**Stack:** Electron, React (same components as web), Tailwind CSS, Recharts

**Key Features:**
- **Dashboard** — Live GPU utilization chart, stats cards, start/stop toggle
- **Earnings** — Balance, transaction history, payout request
- **Jobs** — Job history with status, earnings, duration
- **Live GPU Monitor** — Real‑time gauges (utilization, VRAM, temp, fan), process table, container control (pause/resume/stop/restart/logs)
- **Settings** — Pricing, auto‑start, theme, storage
- **System Tray** — Quick actions, earnings popup
- **Login** — API token authentication

**Build:** `electron-builder` produces `.dmg` (macOS), `.exe` (Windows), `.AppImage` (Linux).

---

## 5. Provider Agent

### 5.1 Python Agent (`neuralops-agent/`)

**Purpose:** Lightweight agent installed on GPU provider machines.

**Key Components:**
- `agent/main.py` — Entry point, startup sequence, main loop
- `agent/gpu_detector.py` — GPU detection via pynvml + nvidia‑smi fallback
- `agent/gpu_monitor.py` — Per‑second GPU snapshots (utilization, VRAM, temp, fan, power, clocks, processes)
- `agent/network_tester.py` — Internet speed test (speedtest‑cli + HTTP fallback)
- `agent/docker_manager.py` — Secure Docker container execution with NVIDIA runtime
- `agent/job_executor.py` — Full job lifecycle: pull image → download model/data → run → monitor → report
- `agent/websocket_client.py` — Persistent WebSocket with heartbeat, auto‑reconnect, job dispatch
- `agent/container_controller.py` — Pause/resume/stop/restart/logs for Docker containers
- `agent/auto_updater.py` — Self‑update mechanism with hash verification

**Startup Sequence:**
1. Load/register with server
2. Detect GPU hardware
3. Run network speed test
4. Verify Docker + NVIDIA runtime
5. Run DLPerf benchmark
6. Open persistent WebSocket
7. Enter main loop: heartbeat + job polling + GPU monitoring

**Distribution:** PyInstaller → standalone executable for Linux, macOS, Windows.

---

## 6. API Reference

### 6.1 Authentication

All endpoints require `Authorization: Token <key>` header. WebSocket connections use `?token=<key>` query parameter.

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/auth/register/` | Register new user |
| `POST` | `/api/auth/login/` | Login |
| `POST` | `/api/auth/logout/` | Logout |
| `GET` | `/api/auth/user/` | Current user info |
| `PATCH` | `/api/auth/profile/` | Update profile |

### 6.2 GPU Marketplace

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/gpu/listings/` | Search GPU listings |
| `POST` | `/api/gpu/listings/` | Create listing |
| `POST` | `/api/gpu/orders/` | Place order |
| `GET` | `/api/gpu/jobs/` | My jobs |
| `POST` | `/api/gpu/jobs/{id}/complete/` | Complete job |

### 6.3 Escrow

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/escrow/balance/` | View balance |
| `POST` | `/api/escrow/deposit/` | Deposit |
| `POST` | `/api/escrow/withdraw/` | Withdraw |

### 6.4 Consumer Marketplace

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/marketplace/listings/` | Browse listings |
| `POST` | `/api/marketplace/orders/` | Place order |
| `GET` | `/api/marketplace/jobs/` | My jobs |
| `GET` | `/api/marketplace/balance/` | Balance |
| `GET` | `/api/marketplace/leaderboard/` | GPU leaderboard |

### 6.5 Notes

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/notes/v1/notes/` | List notes |
| `POST` | `/api/notes/v1/notes/` | Create note |
| `GET` | `/api/notes/v1/notes/{id}/` | Note detail |
| `POST` | `/api/notes/v1/notes/{id}/share-link/` | Create share link |
| `POST` | `/api/notes/v1/notes/{id}/upload-hf/` | Upload to HF |

### 6.6 Paper

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/paper/projects/{id}/compile/` | Compile |
| `GET` | `/api/paper/projects/{id}/output/pdf/` | Download PDF |
| `POST` | `/api/paper/projects/{id}/github/push/` | Push to GitHub |

---

## 7. Setup & Installation

### 7.1 Prerequisites

- Python 3.10+
- Node.js 18+ & npm
- Docker + NVIDIA Container Toolkit (for GPU provider agent)
- Redis (for Celery + Channels)
- LaTeX distribution (for paper compilation)

### 7.2 Backend Setup

```bash
cd django_api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Seed reference data
python manage.py seed_gpu_models
python manage.py seed_benchmarks
python manage.py seed_finance
python manage.py seed_reputation
python manage.py seed_leaderboard_data

# Create superuser
python manage.py createsuperuser

# Start server
python manage.py runserver
```

### 7.3 Celery Setup

```bash
# Terminal 1: Worker
celery -A django_api worker -l info -P solo

# Terminal 2: Beat (periodic tasks)
celery -A django_api beat -l info
```

### 7.4 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 7.5 Provider Agent Setup

```bash
cd neuralops-agent
pip install -r requirements.txt
python -m agent.main --api-key YOUR_TOKEN --server http://localhost:8000
```

### 7.6 Provider Desktop App Setup

```bash
cd neuralops-gui
npm install
npm run dev
```

---

## 8. Development Workflow

### 8.1 Branch Strategy

- `main` — Production‑ready code
- `develop` — Integration branch
- `feature/*` — New features
- `fix/*` — Bug fixes
- `release/*` — Release candidates

### 8.2 Commit Convention

```
feat(scope): description
fix(scope): description
docs(scope): description
refactor(scope): description
test(scope): description
chore(scope): description
```

### 8.3 Testing

```bash
# Backend tests
python manage.py test

# Frontend tests
npm test
```

### 8.4 Code Quality

```bash
# Python
black .
isort .
flake8 .

# TypeScript
npx tsc --noEmit
npx eslint src/
```

---

## 9. Deployment

### 9.1 Production Backend

```bash
# Set DEBUG=False in .env
# Use PostgreSQL instead of SQLite
# Use Redis for Channels + Celery
# Serve static files via Nginx
# Run ASGI server (Daphne/Uvicorn) behind Nginx
# Run Celery worker as systemd service
```

### 9.2 Frontend Build

```bash
cd frontend
npm run build
# Serve dist/ via Nginx or CDN
```

### 9.3 Desktop App Build

```bash
cd neuralops-gui
npm run dist:mac      # macOS .dmg
npm run dist:win      # Windows .exe
npm run dist:linux    # Linux .AppImage
```

### 9.4 Agent Build

```bash
cd neuralops-agent
pyinstaller pyinstaller.spec --onefile
# Executable in dist/
```

---

## 10. Contributing

### 10.1 Adding a New Django App

1. `python manage.py startapp new_app`
2. Add `'new_app'` to `INSTALLED_APPS` in `settings.py`
3. Create `models.py`, `views.py`, `serializers.py`, `urls.py`
4. Register URLs in main `urls.py`
5. Run `python manage.py makemigrations new_app`
6. Run `python manage.py migrate`

### 10.2 Adding a New Attention Type

1. Create a class inheriting from `BaseAttention`
2. Register with `@ATTENTION_REGISTRY.register("name")`
3. Implement `forward(hidden_states, attention_mask=None, **kwargs)`
4. Add to frontend `AttentionPicker` component

### 10.3 Adding a New Compressor

1. Create a class inheriting from `BaseCompressor`
2. Register with `@COMPRESSOR_REGISTRY.register("name")`
3. Implement `compress()` returning `(model, stats)`
4. Add to frontend configuration options

### 10.4 Adding a New Frontend Page

1. Create `src/features/{feature}/pages/{PageName}.tsx`
2. Add lazy import to `router.tsx`
3. Add route to `children` array
4. Add navigation link in sidebar

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Django Apps | 22+ |
| Models | 120+ |
| REST Endpoints | 200+ |
| WebSocket Consumers | 8 |
| Celery Tasks | 15+ |
| React Pages | 80+ |
| UI Components | 30+ |
| Attention Views | 6 |
| GPU Marketplace Apps | 7 |

---

*NeuralOps — Decentralized GPU Compute for ML Research*
