# NeuralOps Web UI Specification & Architecture Guide

**Version:** 2.0.6  
**Package:** `neuralops.ui`  
**Author:** Taha Majlesi  

---

## 1. Overview & Architecture

The NeuralOps Web UI is a minimal, publication-grade web dashboard engineered to visualize empirical benchmark ledgers, explore registered attention mechanisms and model compressors, inspect system health and invariants, and deploy self-contained static site bundles.

### Technology Stack
- **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/) with asynchronous ASGI handling via [Uvicorn](https://www.uvicorn.org/).
- **Dynamic Interactions:** Local vendored [HTMX](https://htmx.org/) (48 KB) for partial page updates, debounced search, pagination, and real-time ledger polling without full-page reloads.
- **Visualizations:** Local vendored [Observable Plot](https://observablehq.com/plot) (ESM) delivering interactive log-log Roofline models and Pareto frontier scatter charts.
- **Styling:** Custom lightweight dark-mode CSS (`ui.css`, <200 lines) with CSS custom properties, responsive data tables, stat cards, and status badges.
- **Persistence:** Direct parameterized queries over `sqlite3` without ORM overhead.
- **Offline & Zero-CDN Guarantee:** Zero runtime dependencies on external CDNs (fonts, scripts, stylesheets are 100% vendored locally).

---

## 2. CLI Interface

The UI subpackage is integrated into the canonical NeuralOps CLI router (`python3 -m neuralops ui`):

```bash
# Launch local server (default: http://127.0.0.1:8080)
python3 -m neuralops ui

# Launch on custom port without auto-opening browser
python3 -m neuralops ui --port 8088 --no-browser

# Launch in remote Share Mode via Cloudflare Tunnel
python3 -m neuralops ui --share

# Enable HTTP Basic Authentication (required for public deployments)
python3 -m neuralops ui --share --auth admin:securepassword

# Pre-render complete static HTML distribution bundle (for HuggingFace Spaces or static hosting)
python3 -m neuralops ui --static ./dist_site
```

### CLI Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port` | `int` | `8080` | Port for the Uvicorn server to bind to. |
| `--host` | `str` | `127.0.0.1` | Network interface for the server. |
| `--share` | `flag` | `False` | Launches ephemeral Cloudflare quick-tunnel (`https://*.trycloudflare.com`). |
| `--auth` | `str` | `None` | Enforces HTTP Basic Auth (`USERNAME:PASSWORD`) across all endpoints except `/health`. |
| `--static` | `str` | `None` | Builds pre-rendered static site into specified directory and exits. |
| `--no-browser`| `flag` | `False` | Disables automated opening of default web browser upon launch. |

---

## 3. Web Pages & Endpoints

### Core Interactive Routes
- **`GET /` (Dashboard):** Operational overview displaying live invariant counts, active git branch and commit SHA, working tree cleanliness, and live-polling recent benchmark runs.
- **`GET /attentions`:** Searchable catalog of all 131 attention operators with family filtering.
- **`GET /attentions/{name}`:** Deep-dive detail view for individual attention operators with module paths, docstrings, and complete historical ledger executions.
- **`GET /compressors`:** Searchable catalog of all 72 model compressors with compression paradigm filtering.
- **`GET /compressors/{name}`:** Deep-dive detail view for individual compressors with surgery specifications and execution history.
- **`GET /benchmarks`:** Searchable master ledger explorer supporting module name substring search, execution status filter (`active`, `verified`), family filter, pagination, and direct CSV export (`/benchmarks/export`).
- **`GET /models`:** Model Zoo catalog summarizing calibrated models (RoBERTa-base, DistilBERT, Qwen2.5, GPT-2) evaluated across downstream datasets.
- **`GET /notebooks`:** Catalog of 94 canonical demonstration tutorials with instant keyword search.
- **`GET /doctor`:** System diagnostics verifying all 5 canonical repository invariants (`131 / 72 / 58 / 141 / 94`) and SQLite table integrity.

### Partial & API Endpoints
- **`GET /partials/recent-runs`:** Auto-polled every 10s by Dashboard for live telemetry.
- **`GET /partials/benchmarks`:** Dynamic table swaps on pagination or filter change.
- **`GET /partials/attentions`:** Debounced 250ms search filter fragment.
- **`GET /partials/compressors`:** Debounced 250ms search filter fragment.
- **`GET /api/chart/roofline`:** JSON dataset for operational intensity proxy vs speedup log-log scatter plot.
- **`GET /api/chart/pareto`:** JSON dataset for compression ratio vs retained accuracy Pareto frontier.
- **`GET /health`:** Unauthenticated JSON health probe (`{"status": "ok", "version": "2.0.6"}`).

---

## 4. Static Site Generator (HuggingFace Spaces)

When executed with `--static <DIR>`, `neuralops.ui.static_builder` pre-renders:
1. `index.html` (Dashboard)
2. `attentions.html` + 131 detail files in `attentions/<name>.html`
3. `compressors.html` + 72 detail files in `compressors/<name>.html`
4. `benchmarks.html`
5. `models.html`
6. `notebooks.html`
7. `doctor.html`
8. `manifest.json` metadata index
9. Vendored `static/` asset directory (`ui.css`, `htmx.min.js`, `plot.min.js`, `charts.js`)

Total: **210 pre-rendered static HTML documents**.

---

## 5. Security & Authentication

- **Timing Attack Safety:** Uses `secrets.compare_digest` in `BasicAuthMiddleware` to prevent timing side-channel attacks on credentials.
- **SQL Injection Prevention:** All SQL queries use parameter binding (`?`), and dynamic column/field filters use explicit whitelists.
- **Resource Hygiene:** All SQLite database connections use explicit `try ... finally: conn.close()` to guarantee zero unfinalized handles.
