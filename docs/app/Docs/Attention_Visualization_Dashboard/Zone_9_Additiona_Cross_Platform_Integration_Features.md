# 🔗 Features 39‑50: Complete Implementation Specifications

Below are the exhaustive implementation details for all twelve cross‑platform integration features. Each specification includes the technology stack, architecture, data flow, concurrency model, and integration points — all designed to work with your existing Django backend, React frontend, WebSocket/Channels infrastructure, and the full suite of platform apps.

---

## Feature 39: Attention‑to‑Experiment Pipeline

### 39.1 Core Concept

Every attention visualization in the dashboard has a persistent "Train with This" button. When clicked, the system captures the exact state (model identifier, attention type with parameters, layer/head selection, input text), serializes it into a pre‑filled experiment configuration, and launches the training pipeline. After training completes, the resulting attention patterns are automatically compared with the original visualization that triggered the experiment — closing the loop between exploration and verification.

### 39.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **State Capture** | Zustand `useAttentionContext` store | Serialize current dashboard state into a JSON configuration |
| **Pre‑filled Launcher** | React Router with query parameters | Open `/launcher?model=bert‑base&attn=galerkin&text=...` |
| **Experiment Submission** | `api.post('/experiments/start‑train/', config)` | Submit the training job via existing REST API |
| **Job Monitoring** | WebSocket (`ws://host/ws/experiments/{jobId}/`) | Real‑time status updates using existing Channels infrastructure |
| **Result Comparison** | Canvas API + OffscreenCanvas | Render side‑by‑side attention matrices for comparison |
| **State Management** | `useAttentionContext` + `useExperimentStore` | Synchronize original and trained model states |

### 39.3 Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ ATTENTION DASHBOARD (Zone 2)                                     │
│                                                                  │
│  User views attention pattern → clicks "Train with This"         │
│         │                                                        │
│         ▼                                                        │
│  State Snapshot Captured:                                        │
│  { modelId, attentionType, attentionConfig, inputText,           │
│    selectedLayer, selectedHead, activeView, explorationId }      │
│         │                                                        │
│         ▼                                                        │
│  Experiment Launcher Pre‑filled (Zone 7 integration)             │
│  /launcher?attention_origin=exploration_abc123&model=bert‑base   │
│         │                                                        │
│         ▼                                                        │
│  User confirms → POST /api/experiments/start‑train/              │
│         │                                                        │
│         ▼                                                        │
│  Django backend creates TrainingRun with metadata:               │
│  config.attention_origin_exploration_id = "exploration_abc123"   │
│  config.attention_origin_timestamp = "2026-05-26T10:30:00Z"      │
│         │                                                        │
│         ▼                                                        │
│  Training runs (Celery + GPU marketplace if needed)              │
│         │                                                        │
│         ▼                                                        │
│  On completion → WebSocket notification to dashboard             │
│  Dashboard fetches attention patterns from trained model         │
│         │                                                        │
│         ▼                                                        │
│  Comparison View: Original (left) vs. Trained (right)            │
│  Shows divergence heatmap, JSD scores, head‑by‑head delta        │
└─────────────────────────────────────────────────────────────────┘
```

### 39.4 Comparison Engine

After training completes, the system automatically:
1. Fetches the attention patterns from the newly trained model using `POST /api/experiments/attention/` with the same input text.
2. Computes **Jensen‑Shannon Divergence (JSD)** between each corresponding attention head pair. JSD is preferred over KL divergence because it is symmetric and bounded [0,1], making it intuitive for visualization.
3. Computes **head‑by‑head delta matrices** (difference between original and trained attention weights).
4. Renders a **comparison heatmap** with a dedicated color scale: blue = original higher, red = trained higher, white = no difference.
5. Generates a **comparison report** summarizing:
   - Which heads changed most/least
   - Whether attention patterns became more or less diffuse
   - Whether head specialization increased
   - Link to the training run for full details

### 39.5 URL‑Based State Restoration

Every "Train with This" action generates a shareable URL that encodes the full dashboard state:

```
/attention?model=bert-base&attn=galerkin&rank=64&layer=5&head=7&text=The%20quick%20brown%20fox
```

When another researcher opens this URL, the dashboard restores the exact state, allowing them to verify results or launch their own experiment with modifications.

---

## Feature 40: Attention Pattern Knowledge Graph Navigator

### 40.1 Core Concept

Every attention pattern visualization becomes a first‑class node in the platform's knowledge graph, connected to the model that produced it, the experiment that generated it, the researcher who published it, the notes that discuss it, the papers that reference it, the collections that include it, and the community discussions about it. Users navigate this graph through an interactive D3 force‑directed visualization that reveals the web of relationships around any pattern.

### 40.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Graph Layout** | D3.js `d3‑force` simulation (v7.9+) | Physics‑based node positioning with custom force functions |
| **React Integration** | `react‑d3‑graph` or custom hook with `useRef` + `useEffect` | Bridge between React component lifecycle and D3 imperative API |
| **Data Fetching** | React Query (`useQuery`) with pagination | Load graph data incrementally as user expands nodes |
| **Node Rendering** | SVG `<circle>` + `<text>` elements with CSS transitions | Hardware‑accelerated rendering for up to 1000 nodes |
| **Edge Rendering** | SVG `<line>` or `<path>` with `stroke‑width` proportional to relationship strength | Visual encoding of connection importance |
| **3D Alternative** | Three.js via `react‑three‑fiber` (`@react‑three/fiber`) | Optional immersive 3D graph for complex networks |
| **Backend Graph API** | Django REST API endpoint `/api/knowledge‑graph/?node_id=...&depth=2` | Fetch graph neighborhood efficiently |

### 40.3 Graph Data Model

The knowledge graph is built from the existing `NoteLink` model and extended with new relationship types:

```
Node Types:
  - attention_pattern (id, model, layer, head, attention_type)
  - model_version (id, name, version, architecture)
  - experiment (id, name, status, created_at)
  - researcher_note (id, title, author)
  - paper (id, title, source, source_id)
  - collection (id, name, curator)
  - community_discussion (id, title, question_id)

Edge Types:
  - PRODUCED_BY (pattern → model)
  - GENERATED_FROM (pattern → experiment)
  - DISCUSSED_IN (pattern → note)
  - REFERENCED_BY (pattern → paper)
  - COLLECTED_IN (pattern → collection)
  - PUBLISHED_BY (pattern → researcher)
  - FORKED_FROM (pattern → pattern)
  - COMPARED_WITH (pattern → pattern)
  - SIMILAR_TO (pattern → pattern, weight = JSD similarity)
```

### 40.4 Force Simulation Configuration

The D3 force simulation uses five force types for optimal layout:

| Force Type | Configuration | Purpose |
|------------|--------------|---------|
| `d3.forceLink()` | Distance = f(relationship_strength) | Connected nodes attract |
| `d3.forceManyBody()` | `strength(-300)` | All nodes repel each other |
| `d3.forceCenter()` | Center of viewport | Keep graph centered |
| `d3.forceCollide()` | `radius(30)` | Prevent node overlap |
| `d3.forceX()` / `d3.forceY()` | Group by node type category | Positional grouping for visual clarity |

### 40.5 Interactive Features

- **Expand on Click**: Clicking a node reveals its direct neighborhood (depth=1), fetching additional data via the API.
- **Hover Tooltip**: Shows node details (model name, attention type, head number, relationship summary).
- **Drill‑Down**: Double‑click a pattern node to open it in the full attention visualization dashboard.
- **Filter by Relationship Type**: Toggle which edge types are visible (e.g., show only "FORKED_FROM" and "COMPARED_WITH").
- **Search Highlight**: Type a model name or researcher name to highlight all matching nodes.
- **Export**: Export the visible subgraph as SVG, PNG, or JSON for external analysis.
- **Time Slider**: Filter nodes by creation date to see how the knowledge graph evolved over time.

### 40.6 Performance Optimizations

- **Virtualization**: For graphs with >500 nodes, use WebGL rendering (Three.js) instead of SVG DOM elements.
- **Level‑of‑Detail (LOD)**: At zoomed‑out view, nodes are simplified (dot only). Zooming in reveals labels and details.
- **Incremental Loading**: Load graph in layers (depth 0, then depth 1, then depth 2) to avoid overwhelming the initial view.
- **Web Worker**: Run force simulation in a Web Worker to avoid blocking the main thread during layout computation.

---

## Feature 41: Attention API (Programmatic Access)

### 41.1 Core Concept

A public REST API and Python SDK that enables programmatic access to attention visualizations, pattern analysis, and automated pipelines. AI agents (like the platform's research agent) can query attention patterns, automated analysis pipelines can run attention diagnostics on models, and third‑party tools can embed attention visualizations. This follows Neuronpedia's model of providing "a suite of free APIs/tools that accelerate white‑box understanding and experimentation of neural networks."

### 41.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **REST API** | Django REST Framework (DRF) | Standard REST endpoints with Token authentication |
| **API Documentation** | `drf‑spectacular` with Swagger UI / ReDoc | Auto‑generated OpenAPI 3.0 schema |
| **Python SDK** | Custom `neuralops‑attention` pip package | High‑level Python client for the API |
| **Rate Limiting** | Django REST Framework throttling | Prevent abuse (1000 req/hour for free tier, 10000 for pro) |
| **API Versioning** | URL‑based versioning (`/api/v1/attention/`, `/api/v2/attention/`) | Backward compatibility |
| **Authentication** | DRF Token Authentication (same as rest of platform) | Consistent auth across all endpoints |
| **SDK Async Support** | `httpx` with async/await | Non‑blocking API calls for high‑throughput pipelines |

### 41.3 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/attention/analyze/` | Compute attention patterns for a given model, text, and attention type |
| `GET` | `/api/v1/attention/patterns/{pattern_id}/` | Retrieve a specific attention pattern by ID |
| `GET` | `/api/v1/attention/patterns/` | List attention patterns with filtering (model, layer, head, attention_type) |
| `GET` | `/api/v1/attention/compare/` | Compare two attention patterns (JSD, delta matrix) |
| `GET` | `/api/v1/attention/stats/{pattern_id}/` | Get statistics (entropy, concentration, importance, diversity) |
| `POST` | `/api/v1/attention/ablate/` | Simulate head ablation and return predicted output changes |
| `GET` | `/api/v1/attention/leaderboard/` | Get ranked attention patterns by community score |
| `GET` | `/api/v1/attention/models/` | List available models with supported attention types |
| `GET` | `/api/v1/attention/export/{pattern_id}/` | Export pattern as SVG, PNG, JSON, or CSV |

### 41.4 Python SDK Design

The SDK `neuralops‑attention` provides a clean, idiomatic Python interface:

```
from neuralops_attention import AttentionClient

client = AttentionClient(api_key="your-token")

# Analyze attention
pattern = client.analyze(
    model="bert-base-uncased",
    text="The quick brown fox jumps over the lazy dog",
    attention_type="galerkin",
    attention_config={"rank": 64},
    layer=5,
    head=7
)

# Get statistics
stats = client.get_stats(pattern.id)
print(f"Entropy: {stats.entropy}, Importance: {stats.importance}")

# Compare two patterns
comparison = client.compare(pattern.id, other_pattern.id)
print(f"JSD: {comparison.jsd}, Max Delta: {comparison.max_delta}")

# Export
client.export(pattern.id, format="svg", output_path="attention.svg")

# Batch analysis
patterns = client.analyze_batch(
    model="bert-base-uncased",
    texts=["Text A", "Text B", "Text C"],
    layers=[0, 2, 4, 6],
    heads=[0, 4, 8]
)
```

### 41.5 Authentication & Rate Limiting

- **Authentication**: Uses the same DRF Token Authentication as the rest of the platform. Users generate API tokens from their profile settings.
- **Rate Limiting**: Implemented via DRF's `UserRateThrottle` and `AnonRateThrottle`. Free tier: 1000 requests/hour. Pro tier: 10000 requests/hour. Enterprise: unlimited.
- **API Key Management**: Users can create, revoke, and rotate API keys from the `/profile/api‑keys/` page.

### 41.6 Webhook Integration

Users can register webhook URLs that receive notifications when:
- A batch analysis completes
- A new attention pattern is published by a followed researcher
- A pattern receives a certain number of stars or reaches a leaderboard position

Webhooks use HMAC‑SHA256 signatures for verification, following the same pattern as the existing `WebhookSubscription` model in the `notes` app.

---

## Feature 42: Collaborative Attention Workshop (Multi‑User)

### 42.1 Core Concept

Multiple researchers can explore attention patterns together in real time, sharing the same view state, seeing each other's cursors, annotating specific attention connections, discussing in threaded conversations, and recording entire exploration sessions for teaching or review. This transforms the dashboard from a solo analysis tool into a virtual lab meeting platform.

### 42.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Real‑Time Sync** | Y.js (CRDT) via `y‑websocket` provider | Conflict‑free shared state synchronization |
| **WebSocket Server** | Django Channels with Redis channel layer | Persistent connections and pub/sub for room management |
| **Presence/Awareness** | Y.js Awareness API | Track cursor positions, selections, and online status |
| **Annotation System** | Custom overlay on SVG/Canvas | Pin comments to specific attention connections |
| **Session Recording** | `rrweb` (record and replay the DOM) | Capture entire exploration sessions for playback |
| **Voice/Video** | WebRTC via `simple‑peer` or `livekit` | Optional audio/video communication |
| **State Management** | Y.js synced store + Zustand for local UI state | Separate shared state from local UI preferences |

### 42.3 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DJANGO CHANNELS SERVER                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              REDIS CHANNEL LAYER                          │   │
│  │                                                           │   │
│  │  Room: attention_collab_{explorationId}                   │   │
│  │    ├── Channel: user_alice                                │   │
│  │    ├── Channel: user_bob                                  │   │
│  │    └── Channel: user_carol                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ▲                                       │
│          ┌───────────────┼───────────────┐                       │
│          │               │               │                       │
│  ┌───────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐               │
│  │ Alice (React)│ │ Bob (React) │ │Carol (React)│               │
│  │              │ │             │ │             │               │
│  │ Y.js Doc ◄──►│ │Y.js Doc ◄──►│ │Y.js Doc ◄──►│               │
│  │ (shared      │ │(shared      │ │(shared      │               │
│  │  state)      │ │ state)      │ │ state)      │               │
│  └──────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 42.4 Shared State Model (Y.js Document)

The Y.js document contains the following shared types:

```
Y.Map {
  "view": Y.Map {
    "activeView": "head",        // Which visualization view is active
    "selectedLayer": 5,
    "selectedHead": 7,
    "zoomLevel": 1.0,
    "colorMode": "heat"
  },
  "annotations": Y.Array of Y.Map {
    "id": "annotation_123",
    "type": "comment" | "highlight" | "question",
    "position": { "queryTokenIndex": 3, "keyTokenIndex": 7 },
    "content": "This connection might represent syntactic dependency",
    "author": "alice",
    "createdAt": "2026-05-26T10:30:00Z",
    "resolved": false,
    "replies": Y.Array of replies
  },
  "cursorPositions": Y.Map {
    "alice": { "x": 450, "y": 320, "color": "#ff6b6b" },
    "bob": { "x": 680, "y": 410, "color": "#4ecdc4" }
  }
}
```

### 42.5 Collaboration Features

- **Shared View State**: When Alice changes the layer slider, Bob's dashboard updates instantly. The Y.js CRDT handles conflicts automatically.
- **Cursor Presence**: Each participant's cursor appears as a colored dot on the attention heatmap, with their name and avatar. This uses the Y.js Awareness API.
- **Annotation Pins**: Users can click any token pair in the heatmap to drop an annotation pin. The pin appears as a colored marker with a comment. Other users can reply, creating threaded discussions anchored to specific attention connections.
- **"Follow Me" Mode**: One user (the presenter) can enable "Follow Me" mode, which locks other participants' views to match the presenter's navigation. This is useful for teaching or walkthroughs.
- **Session Recording**: The entire exploration session is recorded using `rrweb`, which captures DOM mutations and user interactions. Recordings can be replayed later — useful for asynchronous collaboration (one researcher records an analysis, another watches later).
- **Voice/Video Chat**: Optional WebRTC integration for real‑time discussion during exploration.

### 42.6 Conflict Resolution

Y.js CRDTs provide automatic conflict resolution. When two users simultaneously change the same value (e.g., both move the layer slider), Y.js merges the changes deterministically. For annotations, if two users edit the same comment simultaneously, the CRDT merges characters at the granularity of individual keystrokes.

---

## Feature 43: Polis‑Style Opinion Mapping

### 43.1 Core Concept

For controversial topics in attention research (e.g., "Should attention heads be pruned or kept?"), users submit short statements. Other users vote "agree," "disagree," or "pass." The system uses PCA for dimensionality reduction and k‑means for clustering to map the opinion landscape, visualize consensus groups, and identify "bridging statements" that both groups agree on. This follows Polis's approach of "scaling deliberation by mapping high dimensional opinion spaces."

### 43.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Statement Submission** | Django model + REST API | Store statements and votes |
| **Vote Collection** | React UI with agree/disagree/pass buttons | Gather user opinions |
| **Dimensionality Reduction** | PCA via `scikit‑learn` or `ml.js` (JavaScript) | Reduce high‑dimensional vote matrix to 2D for visualization |
| **Clustering** | Two‑stage k‑means (100 base clusters → 2‑5 group clusters) | Polis's proven approach for opinion grouping |
| **Visualization** | D3.js scatter plot or Plotly.js | Interactive opinion map |
| **Bridging Analysis** | Custom algorithm based on Polis's group‑informed consensus | Identify statements with high agreement across clusters |
| **Backend Processing** | Celery task (async) | Run PCA and clustering asynchronously |

### 43.3 Mathematical Pipeline

The Polis‑style analysis follows a four‑stage pipeline:

**Stage 1: Vote Matrix Construction**
- Each statement becomes a row (N statements)
- Each user becomes a column (M users)
- Each cell is `{agree: 1, disagree: -1, pass: 0}`

**Stage 2: PCA Dimensionality Reduction**
- Apply PCA to the M‑dimensional vote matrix
- Retain the first two principal components (explaining typically 40‑60% of variance)
- This maps each user to a 2D point in "opinion space"

**Stage 3: Two‑Stage K‑Means Clustering**
- Stage A: Run k‑means with k=100 to create fine‑grained base clusters
- Stage B: Run k‑means on the 100 cluster centers with k=2–5, selecting the optimal k via silhouette score
- This produces 2‑5 interpretable opinion groups

**Stage 4: Bridging Statement Identification**
- For each statement, compute the mean vote per group
- Statements where all groups have mean vote > 0.5 (or < -0.5) are "bridging statements" — they represent consensus or shared concern

### 43.4 Visualization

The opinion map renders as:
- **Scatter plot**: Each user is a dot positioned by their first two principal components, colored by group cluster.
- **Group centroids**: Larger markers showing the center of each opinion group.
- **Bridging statements**: Listed in a sidebar, with the statement text and group‑level agreement scores.
- **Statement detail**: Clicking a statement shows a histogram of agree/disagree/pass across all users.
- **Group descriptions**: Auto‑generated labels describing each group's defining characteristics (e.g., "Group A: Supports aggressive pruning" vs. "Group B: Prefers conservative approaches").

---

## Feature 44: Attention‑Based Prompt Engineering Tool

### 44.1 Core Concept

A dedicated tool that uses attention analysis to help researchers debug and improve their prompts. The system highlights which tokens receive the most attention, identifies "attention sinks" (tokens that absorb disproportionate attention, potentially wasting model capacity), compares prompts A/B, and provides improvement suggestions based on attention pattern analysis.

### 44.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Token Attention Analysis** | `contextscope` Python library (2026 release) | Detect attention sinks, lost system prompts, and recency bias |
| **Attention Sink Detection** | Custom algorithm based on ICLR 2025 "When Attention Sink Emerges" | Identify tokens with >2σ above mean attention |
| **Prompt A/B Comparison** | JSD divergence between attention distributions | Quantify how different prompts produce different attention patterns |
| **Suggestion Engine** | LLM‑powered (using platform's research agent) | Generate natural language suggestions based on attention analysis |
| **Visualization** | Custom React heatmap overlay | Highlight attention sinks, low‑attention regions, and prompt structure |
| **Backend API** | `POST /api/attention/analyze‑prompt/` | Accept prompt text, return attention analysis with suggestions |

### 44.3 Attention Sink Detection Algorithm

Based on the ICLR 2025 paper "When Attention Sink Emerges in Language Models":

1. Compute the **mean attention** each token receives across all heads: `mean_attn[i] = Σ_h Σ_q attn[q][i] / (H × N)`
2. Compute the **standard deviation** of mean attention across tokens: `σ = std(mean_attn)`
3. A token is classified as an **attention sink** if: `mean_attn[i] > μ + 2σ`
4. Attention sinks typically occur at:
   - The first token (beginning‑of‑sequence token)
   - Punctuation marks (periods, commas)
   - Special tokens (separator tokens, system prompt boundaries)

### 44.4 Prompt A/B Comparison

Users enter two prompts (Prompt A and Prompt B). The system:
1. Computes attention patterns for both prompts using the same model and attention type.
2. Calculates **per‑token attention divergence** between the two prompts.
3. Highlights tokens where attention differs most.
4. Computes **output probability divergence** (how different are the model's predictions?).
5. Renders a side‑by‑side comparison view with a divergence heatmap between them.

### 44.5 Suggestion Engine

The suggestion engine uses the platform's existing research agent (`agents` app) to analyze the attention patterns and generate natural language suggestions:

- "The token 'not' at position 7 receives 3.2× more attention than the average token. Consider whether negation is critical to your prompt."
- "Your system prompt receives 40% less attention than expected. Consider moving key instructions to the beginning or end of the prompt."
- "The last 10% of your prompt absorbs 35% of total attention (recency bias). If earlier context is being ignored, consider shortening the prompt."
- "The attention sink at position 0 (BOS token) absorbs 25% of total attention. This is normal but may limit capacity for other tokens."

### 44.6 Integration with Platform Apps

- **Prompt Library**: Users can save analyzed prompts to their `prompts` library with attached attention analysis.
- **Experiment Launcher**: "Train with this optimized prompt" button opens the experiment launcher pre‑filled with the improved prompt.
- **Notes**: Auto‑generate a researcher note with the prompt analysis and suggestions.
- **Community**: Share prompt analysis results as community posts for discussion.

---

## Feature 45: Model Diff — Cross‑Architecture Comparison

### 45.1 Core Concept

Load attention patterns from any architecture (Standard MHA, Galerkin, TLinFormer, PADRe, Spectraformer, etc.) for the same input text and visualize how attention differs across architectures. The system computes divergence heatmaps, Jensen‑Shannon Divergence (JSD) scores, head‑by‑head similarity matrices, and structure‑level comparisons.

### 45.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Attention Computation** | Backend computes attention for each model + attention type combination | Generate comparable attention matrices |
| **Divergence Calculation** | JSD via `scipy.spatial.distance.jensenshannon` or custom JS implementation | Compute per‑head divergence |
| **Similarity Matrix** | Cosine similarity between attention pattern vectors | Head‑by‑head comparison |
| **Visualization** | Custom React heatmap + D3 matrix view | Render divergence heatmaps |
| **Backend API** | `POST /api/attention/compare‑architectures/` | Accept model pairs and text, return comparison data |

### 45.3 Comparison Dimensions

| Dimension | Metric | Visualization |
|-----------|--------|---------------|
| **Per‑Head Divergence** | Jensen‑Shannon Divergence (JSD) between corresponding head pairs | Heatmap: rows = layers, columns = heads, color = JSD |
| **Head‑by‑Head Similarity** | Cosine similarity between flattened attention matrices | Similarity matrix: rows = Model A heads, columns = Model B heads |
| **Structural Differences** | Entropy difference, concentration difference, specialization difference | Bar charts comparing per‑layer statistics |
| **Output Impact** | Difference in Logit Lens predictions at each layer | Overlay line chart |
| **Token‑Level Divergence** | Per‑token JSD between the two models' attention distributions | Token‑highlighted text with divergence scores |

### 45.4 Architecture Comparison View

The comparison view presents:
1. **Side‑by‑Side Heatmaps**: Model A (left) and Model B (right) for the same layer/head.
2. **Divergence Heatmap (center)**: A third heatmap showing the absolute difference between the two attention matrices, colored with a diverging scale (blue = Model A higher, red = Model B higher, white = no difference).
3. **Summary Statistics Bar**: At the top, showing overall JSD, per‑layer average JSD, and the head with maximum divergence.
4. **Layer Selector**: A dropdown to switch between layers, with each layer showing its average JSD.

---

## Feature 46: Universal Attention Scoring

### 46.1 Core Concept

Every attention pattern, regardless of which architecture produced it, is scored on four standardized dimensions: Interpretability, Specialization, Efficiency, and Stability. These scores enable cross‑architecture comparison and feed into leaderboards, recommendations, and quality filtering.

### 46.2 Scoring Dimensions

| Dimension | Range | Formula | Interpretation |
|-----------|-------|---------|----------------|
| **Interpretability** | 0‑100 | `100 × (1 − mean_entropy / max_entropy)` | Higher = more focused, easier to interpret |
| **Specialization** | 0‑100 | `100 × (1 − mean_cosine_similarity_to_other_heads)` | Higher = more distinct from other heads |
| **Efficiency** | 0‑100 | `100 × head_importance / mean_attention_weight` | Higher = more information per unit of attention |
| **Stability** | 0‑100 | `100 × (1 − std_across_inputs / mean_across_inputs)` | Higher = more consistent across different inputs |

### 46.3 Computation Pipeline

1. **Entropy**: Computed per head as `H = −Σ p_i log(p_i)` where `p_i` are attention weights.
2. **Cosine Similarity to Other Heads**: For each head, compute cosine similarity of its flattened attention matrix with every other head in the same layer. Take the mean.
3. **Head Importance**: Mean of maximum attention weights across all query positions. This approximates how much the head contributes to the output.
4. **Stability**: Compute attention patterns for 5 different input texts. Measure standard deviation of attention weights at each position. Average across positions.

### 46.4 Score Visualization

The Universal Attention Score is displayed as:
- A **radar/spider chart** showing all four dimensions.
- A **composite score** (weighted average, configurable weights).
- A **percentile ranking** relative to all other patterns of the same architecture.
- A **leaderboard badge** if the pattern ranks in the top 10% for any dimension.

---

## Feature 47: Auto‑Generated Research Paper Section

### 47.1 Core Concept

With one click, generate a complete research paper section (Methods, Results, Discussion, References) based on the current attention analysis. The generated content is a draft that opens in the platform's `paper` LaTeX editor, where the researcher can edit, compile, and submit.

### 47.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Content Generation** | LLM (platform's research agent) with structured prompt | Generate Methods, Results, Discussion sections |
| **Figure Embedding** | SVG export of current attention visualization | Embed figures in the paper |
| **Citation Management** | Platform's `agents` paper library (Semantic Scholar, arXiv) | Auto‑generate BibTeX references |
| **LaTeX Integration** | Platform's `paper` app (Overleaf‑style editor) | Open generated content in the LaTeX editor |
| **Template System** | `NoteTemplate` model (existing) with paper‑specific templates | Provide structured templates for different paper types |

### 47.3 Generation Pipeline

1. **Context Collection**: Gather the current dashboard state (model, attention type, layer, head, text), computed statistics (entropy, concentration, importance), and any community annotations or discussions.
2. **Prompt Construction**: Build a structured prompt for the LLM:
   ```
   Generate a research paper section describing the following attention analysis:
   - Model: {model_name}
   - Attention Type: {attention_type} ({description})
   - Layer: {layer}, Head: {head}
   - Input Text: "{text}"
   - Key Findings: Entropy = {entropy}, Concentration = {concentration}%, Importance = {importance}
   - Community Annotations: {annotations}
   
   Generate three sections:
   1. Methods (describe the model, attention mechanism, and visualization technique)
   2. Results (describe the attention pattern, key observations)
   3. Discussion (implications, limitations, comparison with related work)
   
   Include LaTeX formatting, proper citation commands, and figure references.
   ```
3. **Figure Generation**: Export the current attention visualization as SVG and save it to the `paper` project's assets folder. Generate a LaTeX `\includegraphics` command with appropriate caption.
4. **Reference Collection**: Query the `agents` paper library for papers related to the attention type and model. Generate BibTeX entries.
5. **Content Assembly**: Combine the LLM output, figure references, and bibliography into a complete `.tex` file.
6. **Paper App Integration**: Open the generated content in the platform's `paper` LaTeX editor, ready for editing and compilation.

---

## Feature 48: Particle Flow Attention Visualizer

### 48.1 Core Concept

Attention connections are rendered as flowing particles (like the particle effects in games such as Control or Destiny). Each attention connection is a stream of glowing particles flowing from query token to key token. Particle density represents attention weight, particle speed represents attention "confidence," color represents attention head identity, and particles leave glow trails that fade over time. Users can freeze the flow at any moment to inspect the static pattern.

### 48.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **3D Rendering** | React Three Fiber (`@react‑three/fiber`) + Three.js | WebGL‑based particle system rendering |
| **Particle System** | Custom GLSL shaders (vertex + fragment) | GPU‑accelerated particle simulation |
| **Animation** | `useFrame` hook (R3F) + `requestAnimationFrame` | 60fps animation loop |
| **Post‑Processing** | `@react‑three/postprocessing` (bloom, glow effects) | Visual polish for particle trails |
| **Interaction** | `@react‑three/drei` (OrbitControls, Raycaster) | Camera control and particle selection |
| **Physics** | Simple velocity‑based motion (no full physics engine needed) | Particles follow cubic Bézier curves between tokens |

### 48.3 Particle System Architecture

The particle system uses a **GPU‑based instanced rendering** approach for performance:

1. **Geometry**: Each particle is a small quad (2 triangles) rendered via `InstancedMesh` for efficiency. For 10,000+ particles, this keeps draw calls to 1.
2. **Data Texture**: Attention weights, source positions, target positions, and particle ages are stored in a floating‑point data texture, updated each frame.
3. **Vertex Shader**: Reads the data texture to position each particle along its Bézier curve based on its age.
4. **Fragment Shader**: Computes particle color based on head identity (passed via instance attribute), applies glow effect proportional to attention weight, and fades alpha as particles age.
5. **Particle Lifecycle**:
   - **Birth**: Particles spawn at the query token position at a rate proportional to attention weight.
   - **Travel**: Particles follow a cubic Bézier curve from query to key token over 1‑3 seconds (configurable).
   - **Death**: Particles fade out and are recycled into the pool.

### 48.4 Bézier Curve Computation

Each attention connection defines a cubic Bézier curve:

```
P0 = query token position (x₀, y₀, z₀)
P1 = P0 + (0, arc_height, 0)  (control point 1, raised in y)
P2 = P3 − (0, arc_height, 0)  (control point 2, raised in y)
P3 = key token position (x₃, y₃, z₃)

B(t) = (1−t)³P₀ + 3(1−t)²tP₁ + 3(1−t)t²P₂ + t³P₃
```

The arc height is proportional to the distance between tokens, creating elegant curved trajectories.

### 48.5 Visual Effects

- **Bloom/Glow**: Applied via post‑processing to make particles appear luminous.
- **Trail Effect**: Particles leave a fading trail (stored in a separate render texture, blended with multiplicative fade).
- **Color Mapping**: Each attention head has a distinct HSL color. Particles from the same head share the same hue.
- **Density Encoding**: High‑attention connections have more particles per second, creating denser streams.
- **Freeze Mode**: User presses Space to pause all particles mid‑flight. The frozen particles remain visible as a static snapshot. Pressing Space again resumes the flow.

### 48.6 Performance Considerations

- **LOD (Level of Detail)**: At zoomed‑out view, reduce particle count by 50%. At close‑up view, increase to full density.
- **Culling**: Particles outside the camera frustum are not rendered (Three.js handles this automatically for `InstancedMesh`).
- **Particle Budget**: Maximum 50,000 particles in the pool. Older particles are recycled when the pool is full.
- **Frame Rate Target**: 60fps on mid‑range hardware (GTX 1060 equivalent). Fall back to 30fps on low‑end hardware.

---

## Feature 49: Attention Sonification

### 49.1 Core Concept

Each attention head has a unique sound signature based on its attention pattern. High attention weight produces higher pitch, diffuse attention produces white noise, positional heads produce rhythmic patterns, and semantic heads produce melodic patterns. Users can "listen" to the model thinking — hearing how attention shifts as they change the input text or navigate between layers. This feature is particularly powerful for accessibility (visually impaired researchers) and for detecting anomalies (a sudden change in sound indicates unusual attention behavior).

### 49.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Audio Synthesis** | Web Audio API (native browser API) | Generate sounds without external libraries |
| **Oscillator Nodes** | `OscillatorNode` (sine, square, sawtooth, triangle) | Generate tonal sounds for attention weights |
| **Noise Generation** | `AudioBufferSourceNode` with white noise buffer | Generate diffuse attention sounds |
| **Gain Control** | `GainNode` | Control volume per head |
| **Spatialization** | `PannerNode` (stereo panning) | Position sounds in stereo space based on head index |
| **Scheduling** | `requestAnimationFrame` + Web Audio clock | Synchronize audio with visualization updates |
| **MIDI Integration** | Web MIDI API (optional) | Output to external synthesizers for professional audio |

### 49.3 Sound Signature Design

| Attention Pattern | Sound Type | Parameters |
|-------------------|------------|------------|
| **High attention** (focused) | Pure sine wave | Frequency = 200‑800 Hz (mapped from attention weight), Gain = attention weight |
| **Low attention** (diffuse) | White noise | Gain = 0.05‑0.2 (mapped from entropy) |
| **Positional head** (attends to adjacent tokens) | Rhythmic pulse | Oscillator at 440 Hz with LFO modulating gain at 2‑4 Hz |
| **Semantic head** (attends to semantically related tokens) | Melodic sequence | Frequency follows a pentatonic scale based on token position |
| **Induction head** (copies previous token) | Echo/delay effect | Original sound with 200ms delayed copy |
| **Delimiter head** (attends to punctuation) | Short percussive click | Short noise burst (50ms) with fast decay |

### 49.4 Implementation Approach

The sonification engine runs in a **Web Worker** to avoid blocking the main thread. It receives attention data via `postMessage` and synthesizes audio using an `OfflineAudioContext` (for rendering to buffer) or a real‑time `AudioContext` (for live playback).

**Real‑Time Mode**:
1. Create an `AudioContext` with 12 `OscillatorNode` instances (one per head) and 1 `AudioBufferSourceNode` for noise.
2. Connect each node through a `GainNode` to the `AudioContext.destination`.
3. On each animation frame, update the oscillator frequency and gain based on the current attention weights.
4. Use `PannerNode` to spatialize heads in stereo: Head 0 = hard left, Head 11 = hard right.

**Offline Mode**:
1. Create an `OfflineAudioContext` with the desired duration.
2. Schedule all audio events using the Web Audio API's precise timing (`start(t)`, `stop(t)`).
3. Render to an `AudioBuffer`, then export as WAV or MP3 for download.

### 49.5 User Controls

The sonification panel in Zone 8 (or a floating panel) provides:
- **Play/Pause** button for audio playback.
- **Volume** slider per head.
- **Solo/Mute** toggle per head.
- **Sound Type** selector (sine, square, sawtooth, noise, melodic).
- **Tempo** control (affects rhythmic head patterns).
- **Export Audio** button (downloads WAV file).
- **MIDI Output** selector (if Web MIDI is available).

---

## Feature 50: Attention Style Transfer Gallery

### 50.1 Core Concept

The same attention data can be rendered in multiple artistic styles: Heatmap (classic), Sankey flow (dynamic), Arc diagram (elegant), Chord diagram (circular), Hive plot (structural), 3D particle flow (immersive), Minimalist (for papers/publications), and Cyberpunk neon (for presentations). Users can switch between styles with one click and export any style as SVG, PNG, or MP4.

### 50.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **SVG Rendering** | D3.js (heatmap, arc, chord, hive) | Generate publication‑quality vector graphics |
| **Canvas Rendering** | HTML5 Canvas API | Generate raster graphics for complex styles |
| **3D Rendering** | React Three Fiber (`@react‑three/fiber`) | 3D particle flow style |
| **Animation Export** | `MediaRecorder` API + Canvas capture | Export animated styles as MP4 |
| **SVG Export** | `XMLSerializer` + `Blob` API | Export static styles as SVG |
| **PNG Export** | Canvas `toDataURL()` or `toBlob()` | Export static styles as PNG |
| **PDF Export** | `jsPDF` or `@react‑pdf/renderer` | Export for paper submission |

### 50.3 Style Catalog

| Style | Rendering Engine | Best For | Key Visual Encoding |
|-------|-----------------|----------|---------------------|
| **Heatmap** | SVG (D3) | Detailed analysis, publications | Color intensity = attention weight, cell grid |
| **Sankey Flow** | SVG (D3‑sankey) | Understanding information flow | Flow width = attention magnitude |
| **Arc Diagram** | SVG (D3) | Elegant overview | Arc thickness = attention, arc height = head index |
| **Chord Diagram** | SVG (D3‑chord) | Circular relationship view | Chord width = attention, color = head |
| **Hive Plot** | SVG (D3) | Structural comparison | Axis = token position, link = attention |
| **3D Particle Flow** | WebGL (R3F) | Immersive exploration | Particle density = attention, color = head |
| **Minimalist** | SVG | Papers, presentations | Grayscale, thin lines, clean typography |
| **Cyberpunk Neon** | SVG + CSS glow | Presentations, social media | Neon colors, glow effects, dark background |

### 50.4 Style Transfer Architecture

The gallery uses a **strategy pattern** where each style is a separate renderer component that accepts the same data interface:

```typescript
interface AttentionData {
  tokens: string[];
  matrix: number[][];  // attention weights
  metadata: {
    modelId: string;
    attentionType: string;
    layer: number;
    head: number;
  };
}

interface StyleRenderer {
  id: string;
  name: string;
  description: string;
  render: (data: AttentionData, container: HTMLElement) => void;
  exportSVG: (data: AttentionData) => string;
  exportPNG: (data: AttentionData) => Promise<Blob>;
}
```

Each renderer implements this interface, making it easy to add new styles without modifying existing code.

### 50.5 Export Pipeline

**SVG Export**:
1. Render the attention data using the selected style's SVG renderer.
2. Serialize the SVG element using `new XMLSerializer().serializeToString(svgElement)`.
3. Create a `Blob` with MIME type `image/svg+xml`.
4. Trigger download via a temporary `<a>` element with `download` attribute.

**PNG Export**:
1. Render the SVG to a Canvas element using `canvg` or native `Image` + `drawImage`.
2. Convert Canvas to Blob via `canvas.toBlob()`.
3. Trigger download.

**MP4 Export** (for animated styles like 3D Particle Flow):
1. Use `MediaRecorder` API with `canvas.captureStream(60)` to capture 60fps video.
2. Record for the desired duration (default: 10 seconds).
3. On stop, create a Blob from the recorded chunks.
4. Trigger download.

### 50.6 Integration with Paper App

The Minimalist style is specifically optimized for academic papers:
- Generates publication‑quality vector graphics with proper font sizing.
- Exports as PDF (via `jsPDF`) with embedded vector graphics.
- Auto‑generates a LaTeX `\includegraphics` command.
- Provides pre‑written figure captions describing the attention pattern.

---

## Summary: Implementation Priority & Dependencies

| Feature | Priority | Key Dependencies | Estimated Effort |
|---------|----------|-----------------|------------------|
| 39: Experiment Pipeline | P0 | Existing experiment launcher, WebSocket, Canvas API | 3 weeks |
| 40: Knowledge Graph | P1 | D3.js, existing NoteLink model, Django REST API | 4 weeks |
| 41: Public API + SDK | P2 | DRF, `drf‑spectacular`, `httpx` | 3 weeks |
| 42: Collaborative Workshop | P1 | Y.js, Django Channels, Redis, `rrweb` | 5 weeks |
| 43: Polis Opinion Mapping | P2 | `scikit‑learn`, Celery, D3.js | 3 weeks |
| 44: Prompt Engineering | P1 | `contextscope`, existing agents app | 3 weeks |
| 45: Model Diff | P1 | JSD computation, existing attention registry | 2 weeks |
| 46: Universal Scoring | P2 | Existing benchmark infrastructure | 2 weeks |
| 47: Paper Generator | P1 | Existing paper app, LLM agent | 3 weeks |
| 48: Particle Flow | P2 | React Three Fiber, GLSL shaders | 4 weeks |
| 49: Sonification | P3 | Web Audio API | 3 weeks |
| 50: Style Gallery | P2 | D3.js, Canvas API, existing visualization views | 3 weeks |

All features are designed to be built incrementally on top of the existing platform, reusing as much infrastructure as possible (Django REST API, WebSocket/Channels, React component library, Zustand state management, Celery task queue). Each feature connects to at least two other platform apps, creating a deeply integrated ecosystem.