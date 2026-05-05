# NeuralOps Performance Optimization Plan 🚀

This document outlines the high-impact performance improvements for the Transformer Compression Framework, specifically targeting the Django-React stack.

---

## 1. Phase I: Backend Registry Caching 🧠

Currently, the `Registry` system performs repetitive file system introspection and dynamic imports on every API request to `/api/attentions/` and `/api/compressors/`.

- **Problem:** O(N) directory scanning where N=140+ methods.
- **Solution:**
  - Implement `functools.lru_cache` on registry retrieval methods.
  - Integration of Django's `cache` framework (In-memory/Redis) for system-wide configuration data.
- **Target Files:**
  - [Codes/framework/attentions/core/registry.py](Codes/framework/attentions/core/registry.py)
  - [Codes/django_api/api/views.py](Codes/django_api/api/views.py)
- **Expected Gain:** API response time reduction from ~150ms to <5ms.

## 2. Phase II: Frontend UX & Rendering Isolation ⚡

The real-time telemetry updates (CPU/GPU/RAM) currently trigger shallow re-renders across the entire Dashboard page due to central state polling.

- **Problem:** UI micro-stutters during heavy model training sessions.
- **Solution:**
  - **Hardware Context Provider:** Move hardware polling into a dedicated React Context with a separate update cycle.
  - **React.memo:** Memoize heavy grid components (`AttentionPicker`, `ExperimentTable`).
  - **Throttled Listeners:** Set up a 3000ms hardware polling window with a "dirty-check" to avoid DOM updates if values haven't changed by >1%.
- **Target Files:**
  - [Codes/frontend/src/components/SystemResourcesWidget.tsx](Codes/frontend/src/components/SystemResourcesWidget.tsx)
  - [Codes/frontend/src/pages/DashboardPage.tsx](Codes/frontend/src/pages/DashboardPage.tsx)
- **Expected Gain:** 60FPS UI stability even during 100% CPU/GPU saturation.

## 3. Phase III: Code Splitting & Asset Optimization 🌳

The "NeuralOps" dashboard imports heavy charting libraries (Plotly) and large UI components synchronously in the main bundle.

- **Problem:** Large initial JavaScript payload delay.
- **Solution:**
  - **React.lazy/Suspense:** Dynamically load `LiveLossPlot` and `LiveLogViewer` only when the tab is active.
  - **Dynamic Imports:** Use Vite's dynamic import for large method descriptions in the `AttentionPicker`.
- **Target Files:**
  - [Codes/frontend/src/pages/DashboardPage.tsx](Codes/frontend/src/pages/DashboardPage.tsx)
  - [Codes/frontend/vite.config.ts](Codes/frontend/vite.config.ts)
- **Expected Gain:** ~40% reduction in First Contentful Paint (FCP).

## 4. Phase IV: Hardware-Aware Background Tasks 🏗️

Synchronizing backend processing with frontend logging without blocking the main event loop.

- **Problem:** API latency spikes when long-running evaluations start.
- **Solution:**
  - **Subprocess Decoupling:** Ensure `runners.py` uses detached processes for evaluation tasks.
  - **WebSocket Buffer:** Implement a server-side buffer for logs to prevent flooding the frontend socket.
- **Target Files:**
  - [Codes/django_api/api/runners.py](Codes/django_api/api/runners.py)
  - [Codes/django_api/api/consumers.py](Codes/django_api/api/consumers.py)
- **Expected Gain:** Zero-latency API responsiveness during active experiments.

---

### Implementation Priority

1. **Registry Caching** (Immediate impact, low risk)
2. **Telemetry Isolation** (Improves usability)
3. **Code Splitting** (Improves load time)
