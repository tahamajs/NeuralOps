# 🌐 Efficient Transformer Toolkit – API Documentation

The **Efficient Transformer Toolkit API** provides a comprehensive REST and WebSocket interface for training, evaluating, compressing, and monitoring Transformer models.  
It consists of two main components:

- **Interactive CLI** (`run.py`) – a terminal‑based launcher with questionary menus and a real‑time Matplotlib training dashboard.
- **Django REST API** (`django_api/`) – a production‑ready web API with background job processing, WebSocket live metrics, and artifact management.

Both components share the same underlying framework (`framework/`) and give you full access to 30+ attention mechanisms, 20+ compressors, and all GLUE tasks.

---

## 📁 Project Structure

```
Efficient-Transformer-Toolkit/
├── framework/               # Core library (attentions, compression, data, …)
├── django_api/              # Django REST API
│   ├── manage.py
│   ├── django_api/          # Django project settings, ASGI/WSGI
│   ├── api/                 # DRF views, serializers, WebSocket consumers, runners
│   └── requirements.txt
├── run.py                   # Interactive CLI + live dashboard
├── Codes/cli/               # CLI with YAML config profiles (optional)
└── README.md
```

---

## 🚀 Quick Start – Interactive CLI (`run.py`)

### 1. Install requirements

```bash
pip install questionary rich matplotlib
```

### 2. Run the interactive launcher

```bash
python run.py --interactive
```

A menu will guide you through selecting the model, task, attention type, training method, and hyperparameters.  
During training, a **separate matplotlib window** opens, showing real‑time loss curves that update every logging step.

### 3. Command‑line mode

You can also supply all parameters directly:

```bash
python run.py --model distilbert-base-uncased --task sst2 --attn_type aurora --epochs 5 --batch_size 32
```

Run `python run.py --help` for the complete list of options.

---

## 🌍 Django REST API

The Django API is a self‑contained project inside the `django_api/` folder.  
It provides REST endpoints and WebSocket connections for all operations.

### Setup (first time)

```bash
cd django_api
pip install -r requirements.txt
python manage.py migrate
```

### Start the development server

```bash
python manage.py runserver 0.0.0.0:8000
```

For production, use **Daphne** (ASGI) to support WebSockets:

```bash
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application
```

---

## 📚 REST API Endpoints

All endpoints are prefixed with `/api/`.

### 1. Configuration Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config/attentions/` | List all registered attention types |
| `GET` | `/api/config/models/` | List available base models |
| `GET` | `/api/config/tasks/` | List supported GLUE tasks |
| `GET` | `/api/config/presets/` | List training presets (fast, accurate, …) |

**Example response (`/api/config/attentions/`)**
```json
{
  "attentions": ["full", "aurora", "nova", "minimal_nova", "prism_full", "performer", "lsh", ...]
}
```

### 2. Start a Training Job

```http
POST /api/train/
Content-Type: application/json
```

**Request body**
```json
{
  "model_name": "distilbert-base-uncased",
  "task_name": "mrpc",
  "attention_type": "aurora",
  "compression_rank": 64,
  "epochs": 3,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "method": "baseline"
}
```

**Response**
```json
{
  "id": "a1b2c3d4-...",
  "status": "queued",
  "config": {...},
  "created_at": "2026-05-03T10:38:19Z"
}
```

The job runs in the background. You can monitor it via the WebSocket or the job status endpoint.

### 3. Start an Evaluation Job

```http
POST /api/evaluate/
Content-Type: application/json
```

```json
{
  "model_name": "path/to/trained/model",
  "task_name": "mnli",
  "attention_type": "full",
  "batch_size": 32
}
```

### 4. Start a Compression Job

```http
POST /api/compress/
Content-Type: application/json
```

```json
{
  "model_name": "distilbert-base-uncased",
  "compression_method": "tt",
  "target_rank": 16
}
```

### 5. Job Status & Logs

```http
GET /api/jobs/{job_id}/
```

**Response**
```json
{
  "id": "a1b2c3d4-...",
  "status": "running",
  "config": {...},
  "logs": "Step 100: loss=0.4321\nStep 200: loss=0.3123\n...",
  "created_at": "2026-05-03T10:38:19Z"
}
```

Possible statuses: `queued`, `running`, `completed`, `failed`.

### 6. List All Experiments

```http
GET /api/experiments/
```

Returns a list of all completed (and active) experiments with their configurations and statuses.

### 7. Experiment Artifacts

```http
GET /api/experiments/{job_id}/artifacts/
```

Returns a list of generated files (plots, reports, logs) for a completed experiment.

### 8. Download Trained Model

```http
GET /api/experiments/{job_id}/model/
```

Returns a ZIP archive containing the saved model (`config.json`, `pytorch_model.bin`, `tokenizer.json`, …).

---

## 📡 WebSocket – Real‑Time Metrics

Connect to a job’s WebSocket endpoint to receive live training logs and metrics as they happen.

**WebSocket URL**
```
ws://localhost:8000/ws/jobs/{job_id}/
```

**Messages received by the client**

```json
{
  "type": "log",
  "data": "Step 100: loss=0.4321"
}
```
```json
{
  "type": "metrics",
  "data": {
    "step": 100,
    "loss": 0.4321,
    "eval_loss": 0.5643,
    "accuracy": 0.85
  }
}
```

You can connect using any WebSocket client (JavaScript, Python `websockets`, Postman, etc.).

**JavaScript example**
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/jobs/a1b2c3d4-.../");
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg);
};
```

---

## ⚙️ Configuration Options

The API accepts the same configuration parameters available in the toolkit.  
Key fields for training:

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | str | HuggingFace model identifier (e.g., `distilbert-base-uncased`) |
| `attention_type` | str | Registered attention key (`aurora`, `nova`, `performer`, …) |
| `task_name` | str | GLUE task (`mrpc`, `mnli`, …) or custom dataset path |
| `compression_rank` | int | Rank for low‑rank attention methods (default 64) |
| `epochs` | int | Number of training epochs |
| `batch_size` | int | Per‑device batch size |
| `learning_rate` | float | Learning rate |
| `method` | str | Training method (`baseline`, `lora`, `rdrop`, …) |
| `compression_method` | str | For compression jobs (`tt`, `svd`, `soka`, …) |
| `target_rank` | int | Rank for compression |

For a complete list, see the `framework/config.py` or the CLI help text.

---

## 🔧 Deployment

### Development

- Use the built‑in Django development server: `python manage.py runserver`
- WebSockets work out of the box (in‑memory channel layer).

### Production

- Use **Daphne** or **Uvicorn** as the ASGI server.
- Switch to **Redis** as the channel layer backend for WebSockets (install `channels_redis`).
- Set `DEBUG=False` and a strong `SECRET_KEY` via environment variables.
- Use a reverse proxy (Nginx) to serve static/media files and proxy to the ASGI server.

**Example Daphne command**
```bash
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application
```

**Example Gunicorn + Uvicorn worker (for HTTP only, no WebSockets)**
```bash
gunicorn django_api.wsgi:application -w 4 -b 0.0.0.0:8000
```

---

## 📊 Monitoring & Logging

- All jobs write structured logs to the `Experiment` model (`logs` field) and to the WebSocket.
- After completion, artifacts (plots, model files, metrics JSON) are stored under `MEDIA_ROOT/<job_id>/`.
- Use the `/api/experiments/{job_id}/artifacts/` endpoint to list them.

---

## 📜 References

- [Efficient Transformer Toolkit – Main README](../README.md)
- [Attention modules documentation](../framework/attentions/README.md)
- [Compression methods documentation](../framework/compression/README.md)
- [CLI configuration profiles](../Codes/cli/README.md)

---

*Train, compress, and serve your models through a clean, modern API.*
