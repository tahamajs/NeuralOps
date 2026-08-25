# Router Subsystem (`Codes/framework/router`)

The `router` subsystem provides intelligent dynamic dispatching mechanisms that route incoming requests, sequences, or prompt tokens to specialized attention modules or model variants based on complexity and computational budget.

---

## 🏗 Architecture Overview

- **Router Base Abstraction (`base.py`, `router_registry.py`)**: Common interface `BaseRouter` defining `route()` and `predict()` methods, registered in `ROUTER_REGISTRY`.
- **Keyword Router (`keyword_router.py`)**: Low-latency deterministic pattern and keyword-matching router for domain-specific prompt classification.
- **Machine Learning Router (`ml_router.py`)**: Feature-based classifier (TF-IDF, embedding clustering, shallow neural net) predicting the optimal model or attention rank for a given query length.
- **Ensemble Consensus Router (`ensemble_router.py`)**: Aggregates multi-router confidence scores to output ensemble routing decisions.
- **Router Manager (`router_manager.py`)**: Orchestrates sequence dispatch, fallback routing, and latency-budget constraint enforcement.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `base.py` | `BaseRouter` abstract class defining standard routing interfaces. |
| `router_registry.py` | `ROUTER_REGISTRY` managing dynamic registration of routing algorithms. |
| `keyword_router.py` | Regex and keyword matching router for fast heuristic routing. |
| `ml_router.py` | Embedding and classification-based predictive router. |
| `ensemble_router.py` | Weighted consensus router combining multiple routing strategies. |
| `router_manager.py` | High-level orchestrator handling request forwarding and fallback paths. |

---

## 🚀 Quick Start & Usage

### 1. Dynamic Request Routing
```python
from framework.router.ml_router import MLRouter
from framework.router.router_manager import RouterManager

router = MLRouter(model_path="./checkpoints/router.pt")
manager = RouterManager(router=router, default_target="svd_linear_attention")

decision = manager.route("Analyze the historical trends in this 4000-token financial document.")
print(f"Selected Attention Mechanism: {decision.target_module}")
```

### 2. Registering a Custom Router
```python
from framework.router.base import BaseRouter
from framework.router.router_registry import ROUTER_REGISTRY

@ROUTER_REGISTRY.register("length_based")
class LengthBasedRouter(BaseRouter):
    def route(self, text: str, **kwargs) -> str:
        return "full_attention" if len(text.split()) < 512 else "svd_linear_attention"
```
