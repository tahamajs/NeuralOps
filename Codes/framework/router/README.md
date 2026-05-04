# 🔀 Task Routing Package

Fast, lightweight, and flexible task detection for NLP pipelines.  
The routing package helps you decide **which downstream model or processing path** to use for a given input text.  
It provides rule‑based, trainable, and ensemble routers – all integrated with a central registry and manager.

## 📦 Module Structure

```
framework/routing/
├── __init__.py             # Public API, exports all routers, registry, manager
├── base.py                 # BaseRouter, TrainableRouter, RouteResult
├── keyword_router.py       # KeywordRouter (rule‑based, zero‑shot)
├── ml_router.py            # TfidfRouter, FastTextRouter (trainable)
├── ensemble_router.py      # EnsembleRouter (weighted / majority / stacking / dynamic)
├── router_registry.py      # ROUTER_REGISTRY (singleton)
└── router_manager.py       # RouterManager (caching, metrics, active router)
```

## 🚀 Quick Start

```python
from framework.routing import KeywordRouter, TfidfRouter, EnsembleRouter, RouterManager, ROUTER_REGISTRY

# 1. Use a single router
router = KeywordRouter()
task, scores = router.route("This product is better than the other one")
print(task)   # 'comparative'

# 2. Train and use an ML router
tfidf = TfidfRouter()
tfidf.train(texts, labels)
task, probs = tfidf.route("A factual statement")

# 3. Use the registry to create routers dynamically
router = ROUTER_REGISTRY.create("fasttext_router", vocab_size=5000, num_classes=7)

# 4. Manage multiple routers with RouterManager
manager = RouterManager(default_router=KeywordRouter())
manager.register_router(tfidf, make_active=True)
result = manager.route("What is the capital of France?")
print(result.task, result.confidence, result.inference_time_ms)

# 5. Ensemble of routers
ensemble = EnsembleRouter(routers=[(KeywordRouter(), 0.3), (tfidf, 0.7)])
task, scores = ensemble.route("I think this is great")
```

## 🧠 Router Details

### 1. `KeywordRouter` (zero‑shot, rule‑based)

- Maintains task‑specific keyword dictionaries with weights.
- Uses regex with word boundaries; case‑insensitive by default.
- Returns task with highest weighted sum; includes confidence threshold.

**Built‑in tasks:** `comparative`, `negation`, `sentiment_positive`, `sentiment_negative`, `question`, `factual`, `opinion`.

**Example:**
```python
router = KeywordRouter(min_confidence=0.4)
router.add_keywords("urgent", {"asap": 2.0, "immediate": 1.5})
task, scores = router.route("Please answer asap!")
```

### 2. `TfidfRouter` (TF‑IDF + Logistic Regression)

- Scikit‑learn pipeline: `TfidfVectorizer` → `LogisticRegression`.
- Trainable, fast, interpretable (feature weights).
- Save/load via `joblib`.

**Example:**
```python
router = TfidfRouter(max_features=1000)
router.train(train_texts, train_labels)
router.save("task_router.joblib")
```

### 3. `FastTextRouter` (Neural)

- PyTorch model: Embedding → Mean pooling → 2‑layer FFN.
- Trains with mini‑batch, supports GPU.
- Requires a tokenizer function.

**Example:**
```python
router = FastTextRouter(vocab_size=50000, embedding_dim=128, num_classes=7)
history = router.train_model(texts, labels, tokenizer=lambda s: [ord(c) for c in s], epochs=5)
```

### 4. `EnsembleRouter`

Combines multiple routers. Strategies:

- `weighted_vote` (default): weighted sum of per‑task probabilities.
- `majority_vote`: each router casts one vote.
- `stacking`: meta‑classifier on router output vectors.
- `dynamic`: a selector router decides which sub‑router to use.

**Example:**
```python
ensemble = EnsembleRouter(routers=[(k_router, 0.4), (ml_router, 0.6)], strategy="weighted_vote")
ensemble.set_selector(KeywordRouter())   # for dynamic strategy
```

## 🗂️ Registry and Manager

### `ROUTER_REGISTRY` (singleton)

Register routers automatically via decorator `@ROUTER_REGISTRY.register("name")`.  
All provided routers are already registered.

```python
from framework.routing import ROUTER_REGISTRY
print(ROUTER_REGISTRY.list())   # {'keyword_router': KeywordRouter, ...}
router = ROUTER_REGISTRY.create("tfidf_router")
```

### `RouterManager`

Manages a collection of router instances with caching and metrics.

- **Caching**: MD5‑based cache with configurable size and confidence threshold.
- **Metrics**: tracks call count and total/average inference time per router.
- **Active router**: set a default router to use when `router_name` is omitted.

```python
manager = RouterManager(cache_size=5000)
manager.register_router(KeywordRouter(), name="kw", make_active=True)
res = manager.route("text")
manager.get_metrics()
manager.clear_cache()
```

## 🧪 Extending

### Adding a custom router

1. Subclass `BaseRouter` (or `TrainableRouter` if training required).
2. Implement `route(self, text) -> (pred, scores_dict)` and `can_route(self, text) -> bool`.
3. (Optional) Decorate with `@ROUTER_REGISTRY.register("my_router")`.

```python
from framework.routing import BaseRouter, ROUTER_REGISTRY

@ROUTER_REGISTRY.register("length_router")
class LengthRouter(BaseRouter):
    name = "length_router"
    def route(self, text):
        if len(text) < 20:
            return "short", {"short": 0.9, "long": 0.1}
        return "long", {"short": 0.1, "long": 0.9}
    def can_route(self, text):
        return True
```

### Adding a new task to `KeywordRouter`

```python
router = KeywordRouter()
router.add_keywords("code", {"python": 1.0, "function": 0.8, "def": 1.2})
```

## 📊 Performance Notes

- `KeywordRouter`: ~0.05–0.2 ms per text.
- `TfidfRouter`: ~1–2 ms per text (after training).
- `FastTextRouter`: ~2–5 ms per text on CPU, ~0.5 ms on GPU (with batching).
- `EnsembleRouter`: sum of times of its members.

All routers are thread‑safe (read‑only after training).  
`RouterManager` is also thread‑safe for reads (cache uses dictionary operations; if you need concurrency, add locks).

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.
