# 🔀 Task Routing Package

Fast, lightweight, and flexible task detection for NLP pipelines.  
The routing package helps you decide **which downstream model or processing path** to use for a given input text.  
Three routers are provided – from simple keyword matching to a trainable neural classifier – plus an ensemble that combines them.

## 📦 Modules

| Module | Description |
|--------|-------------|
| `KeywordRouter` | Rule‑based router using comprehensive keyword dictionaries (7 built‑in task types). |
| `MLRouter` | Machine‑learning router (TF‑IDF + Logistic Regression). Trainable on your own data. |
| `FastTextRouter` | Neural router that mimics FastText (embedding → average → FFN). |
| `EnsembleRouter` | Weighted ensemble of multiple routers for robust predictions. |

## 🚀 Quick Start

```python
from routing.keyword_router import KeywordRouter
from routing.ml_router import MLRouter
from routing.ensemble_router import EnsembleRouter
```

### Keyword Router (zero‑shot)

```python
router = KeywordRouter()
task, scores = router.route("This product is better than the other one")
print(task)   # 'comparative'
print(scores) # {'comparative': 2.8, 'negation': 0, 'sentiment': 0.6, ...}
```

### ML Router (trainable)

```python
ml_router = MLRouter()
ml_router.train(train_texts, train_labels)
task, probs = ml_router.route("This is a comparative sentence")
ml_router.save("task_router.joblib")
```

### Ensemble Router

```python
ensemble = EnsembleRouter([
    (keyword_router, 0.3),
    (ml_router, 0.7)
])
task, conf = ensemble.route("I enjoy this movie")
print(f"Task: {task}, confidence: {conf[task]:.2f}")
```

---

## 🧠 KeywordRouter

### How it works

- Maintains a dictionary of task types → weighted keywords.
- Counts occurrences in input text, applying length‑based weighting (longer words carry more weight).
- Returns the task with the highest total score.

### Built‑in task types

| Task | Example keywords |
|------|------------------|
| `comparative` | better, worse, than, more, less, higher, lower |
| `negation` | not, no, never, nobody, "n't", hardly |
| `sentiment` | good, great, awesome, terrible, awful, love |
| `question` | what, why, how, when, where, who, ? |
| `irony` | ironically, sarcasm, "yeah right", "as if" |
| `factual` | research, study, shows, data, statistics |
| `opinion` | think, believe, feel, "in my opinion", I think |

### API

```python
router.route(text)          # -> (task, scores_dict)
router.batch_route(texts)   # -> [task, ...]
router.get_confidence(text) # -> float (0‑1)
router.add_keywords(task, ["new","keywords"])
```

### Customisation

```python
# Provide your own keyword dictionary
my_keywords = {'spam': ['buy now', 'click here', 'free money']}
router = KeywordRouter(keywords=my_keywords, use_weights=True)
```

---

## 🧪 MLRouter

- Uses **TF‑IDF** vectoriser (max 5000 features, unigrams + bigrams) and **Logistic Regression**.
- Fully scikit‑learn compatible – can be saved/loaded with `joblib`.
- Lightweight and fast (milliseconds per prediction).

### API

```python
router.train(texts, labels)   # fit the pipeline, returns self
router.route("text")          # -> (predicted label, probability dict)
router.save("path.joblib")
router.load("path.joblib")
```

---

## 🤖 FastTextRouter

- A tiny neural network: Embedding → Average Pooling → 2‑layer FFN → Softmax.
- Trainable end‑to‑end with PyTorch.
- Suitable when you have many examples and want a learnable representation.

### Architecture

```
Input: token ids (batch, seq_len)
  │
Embedding (vocab_size, emb_dim)
  │
Mean pooling → (batch, emb_dim)
  │
Dense(emb_dim, hidden_dim) + ReLU + Dropout
  │
Dense(hidden_dim, num_classes) → logits
```

### API

```python
model = FastTextRouter(vocab_size=5000, num_classes=7)
logits = model(input_ids)
preds = model.predict(input_ids)   # returns class indices
```

---

## 🔀 EnsembleRouter

- Combines multiple routers with user‑defined weights (automatically normalised).
- Voting can be weighted; final score = sum(weight * router_score).
- Falls back to a default task (`"factual"`) if no router produces a score.

### API

```python
ensemble = EnsembleRouter(routers=[(r1, 0.3), (r2, 0.7)])
task, scores = ensemble.route("text")
```

---

## 📊 Comparison

| Router | Training Data | Speed | Interpretability | Customisation |
|--------|---------------|-------|------------------|---------------|
| KeywordRouter | None | ~0.1ms | Full (keywords) | Add/remove keywords |
| MLRouter | Yes (text+labels) | ~1ms | Weights (LR) | Replace classifier |
| FastTextRouter | Yes (tokenised) | ~2ms | Embedding | Full PyTorch model |
| EnsembleRouter | Depends on members | Sum of members | Depends on members | Any combination |

---

## 🛠️ Extending

### Adding a new router

1. Implement a class with a `route(self, text) -> (pred, scores_dict)` method.
2. Pass it to `EnsembleRouter` or use standalone.

Example:

```python
class LengthRouter:
    def route(self, text):
        return "short" if len(text) < 20 else "long", {"short": 0.8, "long": 0.2}
```

### Creating a custom task type

```python
KeywordRouter.ROUTER_KEYWORDS["urgent"] = ["asap", "urgent", "immediately", "now"]
```

---

## 📁 Package Structure

```
routing/
├── __init__.py
├── keyword_router.py
├── ml_router.py
├── ensemble_router.py
└── README.md
```

---

## 📜 License

Provided as part of the Efficient Transformer Toolkit. Feel free to use and adapt.
