
# NeuralOps Backend – Efficient Transformer Toolkit & Autonomous Research Agent

**A Django REST Framework + Channels + Celery + LangGraph application for:**

- **Experiment management** – training, evaluation, compression of Transformer models  
- **Real‑time WebSocket** updates for job progress and agent chat  
- **Autonomous ML research agent** – paper search, hypothesis generation, experiment launch, monitoring  
- **Modular compression framework** – 50+ compressors (SVD, pruning, quantization, etc.) for any HuggingFace model  
- **Paper library** – search, save, annotate, citation graph  
- **Training monitoring** – automatic overfitting/plateau/divergence detection with alerts  
- **Global search & activity feed** – cross‑entity audit trail  

---

## Architecture

```
┌──────────────┐      HTTP/REST       ┌─────────────────┐
│   Frontend   │ ──────────────────>  │   Django views   │
│  (React/Vue) │ <──────────────────  │  (api + agents)  │
└──────────────┘      JSON            └────────┬────────┘
                                              │
                                      ┌───────┴────────┐
                                      │   Serializers  │
                                      └───────┬────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │        Django ORM             │
                              │  Experiment, TrainingRun,     │
                              │  ChatSession, ResearchPaper,  │
                              │  AgentSession, ...            │
                              └───────────────┬───────────────┘
                                              │
           ┌──────────────────────────────────┼──────────────────────────────┐
           │                                  │                              │
   ┌───────┴───────┐                  ┌────────┴────────┐          ┌────────┴────────┐
   │  Celery       │                  │  Channels       │          │  Activity Log   │
   │  (background  │                  │  (WebSocket)    │          │  (global feed)  │
   │   jobs)       │                  │  real‑time      │          └─────────────────┘
   └───────────────┘                  │  progress       │
                                      └─────────────────┘
```

- **Django + DRF** serves the REST API.
- **Django Channels** (with Redis) handles WebSocket connections for job updates and agent chat.
- **Celery** executes long‑running tasks (training, compression, autonomous agent loops).
- **LangGraph** orchestrates multi‑agent conversations and autonomous research cycles.

---

## Quick Start

### 1. Requirements

- Python 3.10+
- Redis (for Channels + Celery)
- PostgreSQL (recommended) or SQLite for development
- CUDA‑capable GPU for training (optional, the system also works on CPU)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:  
`django`, `djangorestframework`, `channels`, `celery`, `psutil`, `torch`, `transformers`, `semanticscholar`, `pyalex`, `arxiv`, `langchain`, `langgraph`, `langchain_anthropic`, `matplotlib`

### 3. Configure Environment

Set the following environment variables (or use a `.env` file):

```bash
DJANGO_SETTINGS_MODULE=your_project.settings
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=*
DATABASE_URL=sqlite:///db.sqlite3   # or postgres://...
REDIS_URL=redis://localhost:6379/0
MEDIA_ROOT=./media
ANTHROPIC_API_KEY=sk-ant-...        # for agent LLM
# Optional: OPENAI_API_KEY, SEMANTIC_SCHOLAR_API_KEY
```

### 4. Add Apps to INSTALLED_APPS

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    ...
    'rest_framework',
    'rest_framework.authtoken',
    'channels',
    'api',
    'experiment',
    'agents',
    'notes',            # optional
    'jobscheduler',     # optional
    'model_versioning', # optional
]
```

### 5. Migrate Database

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Create Superuser

```bash
python manage.py createsuperuser
```

### 7. Attach Tokenizer to Model (for compressors)

Many compressors require a tokenizer attached to the model object before compression.  
When loading a model, attach it:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.tokenizer = tokenizer
```

### 8. Run the Server

**Development (HTTP only):**

```bash
python manage.py runserver
```

**With WebSockets (ASGI):**

```bash
daphne -b 0.0.0.0 -p 8000 your_project.asgi:application
```

**Start Celery worker:**

```bash
celery -A your_project worker -l info -P threads
```

---

## Authentication

The API uses **Token Authentication** from Django REST Framework.  
Register a user to obtain a token:

```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"secure123"}'
```

Response: `{"token": "abc123...", "username": "alice"}`

Include the token in every request:

```
Authorization: Token abc123...
```

Cookies are also set after login/register for browser clients.

---

## API Endpoints (Complete)

### Authentication & User Management

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/auth/register/` | Create user, return token |
| POST | `/api/auth/login/` | Login, return token |
| POST | `/api/auth/logout/` | Delete token |
| GET | `/api/auth/user/` | Get current user info |
| PATCH | `/api/auth/profile/` | Update first/last name, email |
| POST | `/api/auth/change-password/` | Change password |
| POST | `/api/auth/refresh-token/` | Issue new token |

### Configuration (read‑only, authenticated)

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/config/attentions/` | Available attention types |
| GET | `/api/config/compressors/` | Available compression methods |
| GET | `/api/config/trainers/` | Trainer backends |
| GET | `/api/config/methods/` | Training methods (LoRA, …) |
| GET | `/api/config/models/` | Supported base models |
| GET | `/api/config/tasks/` | Supported GLUE tasks |
| GET | `/api/config/presets/` | Compression presets |
| GET | `/api/config/losses/` | Loss functions |
| GET | `/api/config/optimizers/` | Optimizers |
| GET | `/api/config/schedulers/` | LR schedulers |
| GET | `/api/config/callbacks/` | Training callbacks |
| GET | `/api/config/augmentations/` | Data augmentations |
| GET | `/api/config/search-models/` | Search HuggingFace Hub |
| GET | `/api/config/hpo/` | HPO options |
| GET | `/api/config/distributed/` | Distributed training options |
| GET | `/api/config/tracking/` | Tracking options (MLflow, WandB, …) |
| GET | `/api/config/deployment/` | Deployment options (HuggingFace Hub, Docker) |

### Experiment & Run Management

| Method | URL | Description |
|--------|-----|-------------|
| GET/POST | `/api/experiments/` | List / create experiments |
| GET/PATCH/DELETE | `/api/experiments/{id}/` | Detail experiment |
| POST | `/api/runs/start-train/` | Start training job |
| POST | `/api/runs/start-evaluate/` | Start evaluation |
| POST | `/api/runs/start-compress/` | Start compression |
| GET | `/api/runs/{id}/` | Run detail |
| POST | `/api/runs/{id}/stop/` | Stop running job |
| GET | `/api/runs/{id}/results/` | Get results & artifacts |
| POST | `/api/runs/{id}/save/` | Save results to permanent storage |

### Mega Launcher

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/launch/` | Launch full experiment (train + eval + compress + deploy) |

### Agent Chat (Interactive)

| Method | URL | Description |
|--------|-----|-------------|
| GET/POST | `/api/chats/` | List / create chat sessions |
| GET/DELETE | `/api/chats/{session_id}/` | Detail / delete session |
| GET | `/api/chats/{session_id}/messages/` | List messages |
| POST | `/api/chats/{session_id}/send/` | Send user message, get agent response |
| GET | `/api/chats/{session_id}/summary/` | Conversation summary |
| GET | `/api/chats/{session_id}/export/` | Export conversation as JSON |

### Paper Research & Library

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/papers/search/` | Search across Semantic Scholar, arXiv, OpenAlex |
| POST | `/api/papers/save/` | Save paper to library |
| GET | `/api/papers/saved/` | List saved papers |
| GET | `/api/papers/saved/{id}/` | Paper detail |
| PATCH | `/api/papers/saved/{id}/annotate/` | Annotate paper |
| DELETE | `/api/papers/saved/{id}/delete/` | Remove from library |
| GET | `/api/papers/library/` | Search saved papers |
| POST | `/api/papers/citation-graph/` | Build citation graph |
| POST | `/api/papers/bulk-import/` | Bulk import papers |
| GET | `/api/papers/recommendations/{run_id}/` | Recommendations for a run |
| GET | `/api/papers/recommendations/library/` | Recommendations from saved papers |
| GET | `/api/papers/search-for-run/{run_id}/` | Search papers for run topic |

### Training Analysis

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/training/analyze/` | Analyze a training run (overfitting, plateau, …) |
| GET | `/api/training/analysis/{run_id}/` | Get latest analysis |
| POST | `/api/training/suggest-next/` | Suggest next experiment |

### Training Monitor

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/monitor/start/` | Start autonomous monitor (Celery) |

### Autonomous Agent

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/agents/start/` | Start autonomous research agent |
| POST | `/api/agents/{id}/stop/` | Pause/stop autonomous agent |

### System & Health

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/health/` | Health check |
| GET | `/api/version/` | API version info |
| GET | `/api/system/status/` | CPU, memory, disk |
| GET | `/api/system/gpu/` | GPU info |

### Load Balancer / GPU Allocation (if framework installed)

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/loadbalancer/gpu/status/` | GPU allocator status |
| GET | `/api/loadbalancer/gpu/nodes/` | GPU node details |
| GET | `/api/loadbalancer/scheduler/status/` | Job scheduler status |
| GET | `/api/loadbalancer/scheduler/queue/` | Pending jobs |
| GET | `/api/loadbalancer/scheduler/history/` | Job history |
| GET | `/api/loadbalancer/cluster/nodes/` | Cluster nodes |
| GET | `/api/loadbalancer/cluster/gpu-check/` | GPU availability |

### Drift, Fairness, Robustness

| Method | URL | Description |
|--------|-----|-------------|
| GET/POST | `/api/runs/{id}/drift/` | Drift detection |
| POST | `/api/runs/{id}/fairness/` | Fairness evaluation |
| POST | `/api/runs/{id}/robustness/` | Adversarial evaluation |

### Model Cards & Lineage

| Method | URL | Description |
|--------|-----|-------------|
| GET/POST | `/api/runs/{id}/model-card/` | Generate / push model card |

### Global Search & Activity

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/search/` | Full‑text search across all entities |
| GET | `/api/activity/` | Recent activity feed |

---

## WebSocket Endpoints

| URL pattern | Group | Description |
|-------------|-------|-------------|
| `ws/experiments/{job_id}/` | `job_{id}` | Real‑time job progress (metrics, step, status) |
| `ws/terminal/` | `terminal` | Live log messages from all jobs |
| `ws/agent/{session_id}/` | `agent_{id}` | Live agent conversation updates |

---

## Autonomous Research Agent

The system supports two modes:

### 1. Interactive Chat (`ChatSession`)

- Users can start a chat session and send messages.
- The LangGraph agent calls tools (search papers, configure experiment, etc.) and returns responses.
- Tool calls are recorded as `experiment_card` messages for rich UI rendering.

### 2. Autonomous Agent (`AgentSession`)

- Users define an **objective** (e.g., “Improve MRPC accuracy above 0.89”).
- The agent runs in a **Celery task**, iterating through cycles:
  1. Read memory and tool definitions.
  2. Call LLM to decide next action.
  3. Execute tools, record actions.
  4. Update memory and repeat.
- Runs until max cycles or cost budget is exhausted.
- All actions are logged to `AgentAction` and memory is stored in `AgentMemory`.

**Start an autonomous agent:**

```bash
curl -X POST http://localhost:8000/api/agents/start/ \
  -H "Authorization: Token abc123" \
  -H "Content-Type: application/json" \
  -d '{"objective":"Improve MRPC accuracy above 0.89","metric_name":"eval_accuracy","max_cycles":50}'
```

---

## Training Monitor

The training monitor is a Celery task that checks a run at 25%, 50%, 75%, and 100% of total steps. At each checkpoint it:

1. Analyzes training for overfitting, plateau, divergence.
2. Generates a Markdown report and saves it as a `ResearcherNote`.
3. If divergence is detected and `auto_stop_enabled` is `True`, stops the run.
4. Searches related papers and appends recommendations to the note.

**Start monitoring:**

```bash
curl -X POST http://localhost:8000/api/monitor/start/ \
  -H "Authorization: Token abc123" \
  -H "Content-Type: application/json" \
  -d '{"run_id":"<uuid>","auto_stop_enabled":true}'
```

---

## Framework Integration

The backend leverages the **Efficient Transformer Toolkit** (`framework`) for:

- **Compression** – all registered compressors (A3, AASVD, CUR, TT, SoKA, …) are available via `COMPRESSOR_REGISTRY`.
- **Attention modules** – injected into any HuggingFace model via `inject_attention()`.
- **Model‑agnostic helpers** – `framework/core/architecture.py` provides `detect_architecture`, `get_transformer_layers`, etc.

If the framework is not installed, the API gracefully returns `501 Not Implemented` for those endpoints.

---

## Deployment

### Production Setup

1. **Use PostgreSQL** instead of SQLite.
2. **Use Redis** for Channels and Celery.
3. **Set `DEBUG=False`** and configure `ALLOWED_HOSTS`.
4. **Collect static files**: `python manage.py collectstatic`.
5. **Serve with ASGI** (Daphne, Uvicorn) behind Nginx.
6. **Run Celery worker** as a systemd service.
7. **Mount `MEDIA_ROOT`** to a persistent volume (e.g., NFS).

### Example `asgi.py`

```python
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from api.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(websocket_urlpatterns),
})
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | – | Anthropic API key for agent LLM |
| `OPENAI_API_KEY` | – | OpenAI API key (alternative LLM) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model for agent |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `DATABASE_URL` | `sqlite:///db.sqlite3` | Database URL |
| `SECRET_KEY` | – | Django secret key |
| `MEDIA_ROOT` | `./media` | Artifact storage |
| `SEMANTIC_SCHOLAR_API_KEY` | – | Semantic Scholar API key (optional) |
| `OPENALEX_EMAIL` | – | OpenAlex polite email |

---

## Troubleshooting

**“Framework not installed” errors**  
The API returns `501 Not Implemented` for endpoints that require the `framework` package. Install it or update `PYTHONPATH`.

**WebSocket connection fails**  
Ensure `channels` is installed and ASGI is properly configured. In production, WebSocket upgrades must be allowed by the reverse proxy.

**Training jobs stuck in “queued”**  
Check that Celery worker is running: `celery -A your_project worker -l info`.  
Look at the worker logs for errors.

**Compression reports no parameter reduction**  
Many compressors use soft pruning (zeroing) and do not physically remove parameters. The stats show effective reduction. To get actual memory savings, export the factorized weights.

---

## License

MIT License – see `LICENSE` file.

---

