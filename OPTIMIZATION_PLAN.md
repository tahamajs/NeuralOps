# Performance Improvement Plan (Agent Implementation Guide)

This document outlines the high-priority performance improvements for the ML/NLP experiment platform, categorized by system component.

---

## 1. Backend & Database (Django)

### Query Optimization

- **Problem:** Potential N+1 queries and lack of efficient data fetching.
- **Action:**
  - Use `select_related()` for ForeignKey relationships (e.g., `TrainingRun.experiment`).
  - Use `prefetch_related()` for ManyToMany or reverse ForeignKey relationships (e.g., `Experiment.runs`).
  - Add composite indexes to `TrainingRun` for `(experiment, status)` and `(created_at)`.

- **Target Files:**
  - [Codes/django_api/api/views.py](Codes/django_api/api/views.py)
  - [Codes/django_api/api/models.py](Codes/django_api/api/models.py)

### API Enhancements

- **Action:**
  - Implement cursor-based pagination for large result sets.
  - Enable Gzip compression in `MIDDLEWARE`.
  - Add caching for static configuration endpoints (models, tasks, presets) using `@cache_page`.

- **Target Files:**
  - [Codes/django_api/django_api/settings.py](Codes/django_api/django_api/settings.py)
  - [Codes/django_api/api/views.py](Codes/django_api/api/views.py)

---

## 2. Frontend (React + Vite)

### State & Memory Management

- **Problem:** Infinite growth of log/metric arrays in Zustand store.
- **Action:**
  - Implement bounded log buffers (e.g., max 1000 entries).
  - Implement metric downsampling for long-running jobs to prevent UI lag.

- **Target Files:**
  - [Codes/frontend/src/store/experimentStore.ts](Codes/frontend/src/store/experimentStore.ts)

### Infrastructure & Bundle

- **Action:**
  - Implement code splitting for large modules like Plotly in `vite.config.ts`.
  - Add virtualization (e.g., `react-window`) to the Experiment list and Log viewer.

- **Target Files:**
  - [Codes/frontend/vite.config.ts](Codes/frontend/vite.config.ts)
  - [Codes/frontend/src/pages/](Codes/frontend/src/pages/)

---

## 3. Real-time Communication (WebSockets)

### Message Throttling

- **Problem:** High frequency updates causing browser bottlenecks.
- **Action:**
  - Implement server-side batching/throttling for logs and metrics (e.g., send every 500ms).

- **Target Files:**
  - `Codes/django_api/api/consumers.py`

---

## 4. GPU Scheduler

### Concurrency & Performance

- **Problem:** Lock contention and synchronous resource checking.
- **Action:**
  - Use fine-grained locking (separate queue/GPU locks) in `GPUScheduler`.
  - Implement caching for Job objects to avoid repeated DB hits in the scheduler loop.

- **Target Files:**
  - [Codes/django_api/jobscheduler/scheduler.py](Codes/django_api/jobscheduler/scheduler.py)


---

## Implementation Priority

1. **Tier 1 (Immediate):** Zustand log bounding, Django query optimization (select_related), API config caching.
2. **Tier 2 (Med-term):** WebSocket throttling, GPU scheduler job caching, Vite manual chunks.
3. **Tier 3 (Platform):** UI virtualization, Cursor pagination, Postgres migration.

