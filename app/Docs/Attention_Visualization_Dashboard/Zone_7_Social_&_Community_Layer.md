# 🧠 Zone 7: Social & Community Layer — Complete Implementation Specification

After extensive research across 15+ production systems—including ComparIA (FastAPI + SvelteKit arena architecture), Neuronpedia (microservices with open‑source circuit‑tracer integration), Yupp ($33M a16z blockchain‑incentivized evaluation), Polis (PCA + k‑means consensus visualization), Anthropic Circuit Tracer (open‑source attribution graphs), HeedVision (WebXR collaborative attention‑aware visualizations), Yorkie (CRDT‑based real‑time collaboration), and Liveblocks (sync engine architecture)—here is the **complete, production‑ready implementation specification** for all nine Zone 7 features.

---

## 📐 Overall Architecture for Zone 7

Zone 7 follows a **decoupled micro‑frontend architecture** where each social feature is an independent module that communicates through a shared event bus and centralized state management. The architecture is inspired by ComparIA's "decoupled web application featuring a SvelteKit frontend and a FastAPI backend" and Neuronpedia's "microservices architecture, separating the webapp, database, inference, and auto‑interpretation services."

### Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Frontend Framework** | React 19 with Concurrent Mode | Enables 60FPS rendering during real‑time updates, even with large reaction counts. As noted in research: "利用其最新的 use 钩子和增强的并发模式（Concurrent Mode），显著降低了重绘频率，确保在万人大群消息滚动时依然保持 60FPS 的流畅度。" |
| **State Management** | Zustand + Immer | Lightweight, performant stores for reaction state, voting sessions, and discussion threads. Zustand's immutable update pattern via Immer prevents unnecessary re‑renders. |
| **Real‑time Communication** | WebSocket (Django Channels) + Server‑Sent Events (SSE) | Dual protocol: WebSocket for bidirectional communication (discussions, collaboration), SSE for unidirectional streaming (vote count updates, reaction notifications). ComparIA uses "a hybrid approach combining FastAPI for the web server and REST endpoints with a custom SSE (Server‑Sent Events)" for real‑time streaming. |
| **Collaborative Editing** | Yjs + y‑websocket provider | CRDT‑based conflict‑free synchronization for simultaneous annotation and discussion editing. "True CRDTs underpin frameworks like Yjs and Automerge, powering many collaborative text editors and document tools." |
| **Data Fetching** | TanStack Query (React Query) v5 | Automatic cache invalidation, optimistic updates for reactions/votes, and background refetching. The research confirms: "With React Query, you'll master efficient data fetching." |
| **Backend API** | Django REST Framework + FastAPI microservice | DRF for core CRUD; FastAPI for high‑throughput endpoints (Elo rating computation, feed aggregation). ComparIA's architecture validates this hybrid: "ComparIA uses a hybrid approach combining FastAPI for the web server and REST endpoints." |
| **Graph Visualization** | D3.js 7.9.0 + react-force-graph-2d | For knowledge graph and lineage visualization. The research confirms: "核心采用D3.js 7.9.0实现力导向图布局、节点拖拽和缩放交互，结合React 19.2.3进行组件化开发，使用TypeScript确保类型安全。" |
| **Widget Embedding** | iframe + postMessage API | Isolated sandbox for embedded attention visualizations. "The inter‑frame communication system uses the browser's postMessage API to enable secure, cross‑origin communication between the parent page and the iframe." |
| **Caching** | Redis + IndexedDB (via idb‑keyval) | Redis for server‑side caching of aggregated reaction counts and Elo ratings; IndexedDB for client‑side persistence of draft comments and vote history. |

### Event Bus Architecture

All Zone 7 features communicate through a centralized `AttentionEventBus` (implemented as a Zustand store with middleware):

```typescript
// Conceptual architecture (not implementation code)
interface AttentionEvent {
  type: 'reaction' | 'vote' | 'comment' | 'fork' | 'challenge_submission' | 'collection_update';
  payload: Record<string, unknown>;
  patternId: string;
  userId: string;
  timestamp: number;
}

// Features subscribe to relevant event types
eventBus.subscribe('reaction', updateReactionCounts);
eventBus.subscribe('vote', updateEloRatings);
eventBus.subscribe('comment', notifyDiscussionParticipants);
```

This pattern follows the "decoupled web application" approach validated by ComparIA, where "the backend handles complex logic for LLM orchestration via LiteLLM, session management, and persistent logging."

---

## Feature 25: Multi‑Dimensional Reaction System

### 25.1 Core Concept

Researchers react to attention patterns with context‑rich signals beyond binary like/dislike: 🔬 **Interesting**, ⚡ **Efficient**, 🎯 **Specialized**, 🔄 **Redundant**, 🧩 **Novel**. Each reaction is a distinct data point that contributes to a multi‑faceted "pattern quality score" displayed on every attention visualization and feeding into the recommendation engine.

### 25.2 Backend Implementation

**Database Model (Django ORM, extends existing `reaction` system):**

The model uses a composite unique constraint to prevent duplicate reactions per user per pattern, with a JSON field for extensible metadata.

**API Endpoint:** `POST /api/attention/{pattern_id}/react/`

The endpoint handles both creation and toggling (if user already reacted with the same type, remove it; otherwise, add it). This follows the toggle pattern used by HuggingFace model likes and GitHub star/unstar interactions.

**Aggregation Service (Celery task):**

A periodic Celery task (every 60 seconds) recalculates the aggregate reaction counts and quality scores for patterns that have received new reactions. This avoids expensive real‑time aggregation on every reaction. The task:
1. Queries all patterns with `reactions_updated_at > last_run`
2. Groups reactions by pattern and type
3. Computes weighted quality score: `score = Σ (reaction_count × reaction_weight) / total_reactions`
4. Updates the pattern's cached score in Redis and the database

**Reaction Weights (configurable per community norms):**
| Reaction | Weight | Rationale |
|----------|--------|-----------|
| 🔬 Interesting | +2 | Indicates research value |
| ⚡ Efficient | +3 | Strong signal for compression |
| 🎯 Specialized | +3 | Indicates clear head function |
| 🔄 Redundant | -1 | Negative signal, useful for pruning |
| 🧩 Novel | +4 | Highest value, rare discovery |

### 25.3 Frontend Implementation

**Component:** `ReactionBar` (rendered below every attention pattern in Zone 2)

The component uses React 19's `useOptimistic` hook for instant UI feedback. When a user clicks a reaction, the UI updates immediately (optimistic), the API call is dispatched in the background, and if the API fails, the optimistic update is rolled back. This approach is validated by React 19's "use 钩子和增强的并发模式（Concurrent Mode），显著降低了重绘频率。"

**State Management:** A Zustand store `useReactionStore` holds reaction state per pattern. The store subscribes to the `AttentionEventBus` for real‑time reaction updates from other users (received via WebSocket).

**Real‑time Updates:** When User A reacts to a pattern, the backend broadcasts the event via Django Channels to all other users currently viewing that pattern. The WebSocket message contains `{ pattern_id, reaction_type, user_id, action: 'add' | 'remove' }`. Other clients update their local state immediately.

### 25.4 Integration Points

- **Recommendation Engine (Zone 7, Feature "For You" Feed):** The quality score computed from reactions is one of the primary signals for the personalized recommendation algorithm.
- **Leaderboards (existing `leaderboards` app):** Patterns with the most reactions per type appear on dedicated leaderboards ("Most Interesting Patterns", "Most Novel Discoveries").
- **Portfolio (existing `portfolios` app):** The reactions a user has given and received are displayed on their Attention Portfolio, contributing to their reputation score.
- **Activity Feed (existing `activity` system):** Each reaction generates an activity event visible to the user's followers.

### 25.5 Performance Considerations

- **Debounced aggregation:** Reaction counts are not recomputed on every click. The Celery task runs every 60 seconds, and interim counts are cached in Redis with a 60‑second TTL.
- **Optimistic UI:** The frontend updates immediately; the user never waits for a server response.
- **Batch WebSocket messages:** If a pattern receives many reactions in a short time (e.g., during a live demo), the backend batches WebSocket messages every 500ms to avoid flooding clients.

---

## Feature 26: Attention Arena — Blind Pairwise Voting (ComparIA‑Style)

### 26.1 Core Concept

Two attention visualizations are shown side‑by‑side, **anonymized** (hiding which model/head produced them). Users vote on specific criteria: "Which pattern is more interpretable?" or "Which better captures syntax?" Votes generate **Elo ratings** for attention patterns, enabling a community‑driven ranking system.

The implementation is directly inspired by ComparIA's architecture: "users interact with two anonymous models simultaneously, vote on the better response, and subsequently reveal the models' identities" — generating "high‑quality preference datasets." The Bradley‑Terry model, which "is a statistical basis for ELO‑like ratings," is used with "bootstrap resampling to provide robust confidence intervals."

### 26.2 Backend Implementation

**Database Models:**

1. `ArenaMatch` — Records each pairwise comparison session.
2. `ArenaVote` — Records each individual vote with criterion and winner.
3. `AttentionPatternElo` — Stores current Elo rating and match history for each pattern.

**Elo Rating Algorithm:**

The Elo computation uses the standard formula:
```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R_A_new = R_A + K × (S_A - E_A)
```

Where:
- `R_A`, `R_B` are current ratings (starting at 1500)
- `E_A` is the expected probability of A winning
- `S_A` is the actual outcome (1 for win, 0 for loss, 0.5 for tie)
- `K` is the development factor (32 for new patterns, 16 for established ones)

The implementation follows the `elote` Python library approach: "a powerful Python library for implementing and comparing rating systems ... sports teams, or prioritizing features in your product backlog." The K‑factor decays as the pattern accumulates matches: `K = max(16, 32 / sqrt(1 + matches_played))`.

**Arena Session Management:**

The arena uses a session‑based approach where each user is assigned a unique arena session. The backend selects two patterns for comparison using a smart pairing algorithm:

1. First, select patterns within ±200 Elo of each other (balanced matches provide more information)
2. Prioritize patterns that have fewer total matches (exploration vs. exploitation trade‑off)
3. Ensure the user hasn't already voted on this exact pair
4. Randomly assign which pattern appears on the left vs. right (to avoid position bias)

**API Endpoints:**

- `GET /api/attention/arena/next-pair/` — Returns the next pair of patterns for the user to compare. The response includes anonymized visualization data (SVG/JSON) with model/head metadata stripped.
- `POST /api/attention/arena/vote/` — Submits a vote. Body: `{ match_id, winner_id, criterion }`. The backend updates Elo ratings synchronously within the transaction.
- `GET /api/attention/arena/leaderboard/` — Returns the current Elo rankings, filterable by criterion and model architecture.
- `GET /api/attention/arena/my-stats/` — Returns the user's voting history, accuracy (how often they agree with consensus), and contribution to the dataset.

**Bootstrap Resampling for Confidence Intervals:**

Following ComparIA's approach, the backend periodically (daily) runs bootstrap resampling to compute confidence intervals for Elo ratings. This "provides robust confidence intervals for model positions." The process:
1. Sample the vote history with replacement 1000 times
2. Recompute Elo ratings for each bootstrap sample
3. Report the 2.5th and 97.5th percentiles as the 95% confidence interval

### 26.3 Frontend Implementation

**Arena Page Layout:**

The arena page displays two attention visualizations side‑by‑side in a split‑screen layout. Each visualization is rendered in a sandboxed component that strips identifying information (model name, layer/head numbers replaced with "Pattern A" and "Pattern B").

**Reactive State Management:**

ComparIA's approach is replicated: "The arena interface is powered by a reactive state object arena defined in chatService.svelte.ts." In React, this translates to a Zustand store `useArenaStore` that manages:
- Current match pair
- User's vote history for this session
- Loading states for next pair fetching
- Anonymized visualization data

**Vote Flow:**

1. User arrives at the arena → automatically fetches the first pair
2. Both visualizations render simultaneously with a loading skeleton
3. User examines both patterns (hover, zoom, pan — all standard interactions work)
4. User clicks a vote button below either pattern
5. **Optimistic update:** The UI immediately shows a checkmark on the winner and a subtle animation
6. The API call is dispatched in the background
7. After a brief transition (500ms), the next pair slides in
8. After 5 votes, the user can optionally reveal the identities of the patterns they just voted on

**Reveal Mechanism:** After voting, a "Reveal Identities" button becomes available. Clicking it replaces the "Pattern A/B" labels with the actual model names, attention types, layer/head numbers, and the current Elo ratings. This delayed reveal prevents bias during voting.

### 26.4 Integration Points

- **Leaderboards (existing `leaderboards` app):** Elo ratings are displayed on the Attention Pattern Leaderboard, with weekly/monthly/all‑time rankings.
- **Recommendation Engine:** High‑Elo patterns are prioritized in the "For You" feed.
- **Reputation System (existing `gpu_reputation` app):** Users earn reputation points for voting consistently with consensus. Inversely, users who consistently vote against consensus have their vote weight reduced.
- **Dataset Generation:** Arena votes are exported as a preference dataset (JSONL format) for research use, following ComparIA's model of "collecting real‑world interaction data."

### 26.5 Anti‑Gaming Measures

- **Minimum account age:** Users must have been registered for at least 24 hours and have completed at least 5 attention explorations before voting.
- **Vote consistency scoring:** Users who always vote for the left option (position bias) are flagged and their votes are weighted lower.
- **Time‑based filtering:** Votes cast in under 2 seconds are discarded (insufficient time to evaluate).
- **CAPTCHA on rapid voting:** If a user submits more than 10 votes per minute, a CAPTCHA challenge is triggered.

---

## Feature 27: Fork & Remix — Lineage Tracking

### 27.1 Core Concept

Every published attention visualization can be **forked** by another researcher, maintaining full lineage tracking. The implementation follows Neuronpedia's model where the platform "fully supports uploading your own graphs, regardless of whether or not you use the circuit‑tracer library, your own library, or different models." GitHub's fork model is adapted for attention visualization: each fork creates a new exploration with the original's configuration, maintains a parent pointer, and visualizes the lineage as a directed acyclic graph.

### 27.2 Backend Implementation

**Database Models:**

1. `AttentionFork` — Links a forked exploration to its parent.
2. `ForkChain` — Denormalized table storing the full lineage path for efficient querying: `{ exploration_id, ancestor_id, depth }`.

**Lineage Chain Query:**

Using Django's recursive CTE (Common Table Expression) support, the full lineage is retrieved in a single query:

```sql
WITH RECURSIVE lineage AS (
    SELECT id, parent_id, creator_id, 0 AS depth
    FROM attention_fork
    WHERE id = {fork_id}
    UNION ALL
    SELECT af.id, af.parent_id, af.creator_id, l.depth + 1
    FROM attention_fork af
    JOIN lineage l ON af.id = l.parent_id
)
SELECT * FROM lineage ORDER BY depth;
```

This returns the complete ancestor chain, enabling the lineage tree/graph visualization.

**Fork Operation:**

When a user clicks "Fork," the backend:
1. Creates a new `AttentionExploration` record with the same configuration (model, attention type, text, layer/head)
2. Creates an `AttentionFork` record linking the new exploration to the original
3. Copies any annotations/discussions from the original (optional, user‑selectable)
4. Generates a fork notification for the original creator

### 27.3 Frontend Implementation

**Fork Button:** A prominent "Fork" button (⑂ icon, like GitHub) appears in the pattern toolbar. Clicking it opens a modal with options:
- Fork name (auto‑generated: "fork of [original_name]")
- What to copy: configuration only, or configuration + annotations
- Visibility: public / team / private

**Lineage Graph Visualization:**

The lineage is visualized using `react-force-graph-2d` (a React wrapper around D3.js force simulation). The research confirms this approach: "核心采用D3.js 7.9.0实现力导向图布局、节点拖拽和缩放交互，结合React 19.2.3进行组件化开发，使用TypeScript确保类型安全。"

The graph shows:
- **Nodes:** Each exploration as a colored circle (color = attention type, size = number of reactions)
- **Edges:** Fork relationships as directed arrows
- **Current node:** Highlighted with a glowing border
- **Root node:** The original exploration marked with a star icon
- **Interactions:** Click a node to navigate to that exploration, hover to see metadata (creator, date, fork count), drag to rearrange

**Graph Component Implementation:**

The component uses `react-force-graph-2d` with the following configuration:
- Physics simulation: d3‑force with centering, collision detection, and link distance proportional to time between forks
- Zoom: mouse wheel with limits (0.5x to 3x)
- Pan: drag on empty space
- Node color: mapped from attention type color palette
- Node size: `5 + (reaction_count * 0.5)` pixels
- Link width: `1 + (depth * 0.5)` pixels (thicker for deeper lineages)

### 27.4 Integration Points

- **Attention Portfolio (Feature 33):** Forked explorations appear on the creator's portfolio with "Forked by X users" count.
- **Activity Feed (existing `activity` system):** Fork events are broadcast to followers of both the original creator and the forker.
- **Knowledge Graph (future Zone 7 feature):** The lineage graph is integrated into the larger platform knowledge graph, connecting explorations, models, and researchers.

---

## Feature 28: Attention Challenge Creator & Leaderboard

### 28.1 Core Concept

Users create challenges for the community, the community submits attention patterns as entries, votes determine winners, and badges/platform credits are awarded. This feature combines elements of Kaggle competitions, SciArena's community‑driven evaluation ("collected over 20,000 votes from human researchers across diverse scientific domains"), and ComparIA's preference dataset generation.

### 28.2 Backend Implementation

**Database Models:**

1. `AttentionChallenge` — Challenge definition with title, description, criteria, deadline, prize pool.
2. `ChallengeSubmission` — Links an attention exploration to a challenge.
3. `ChallengeVote` — Community votes on submissions.
4. `ChallengeBadge` — Badges awarded to winners and participants.

**Challenge Lifecycle:**

1. **Draft:** Creator defines the challenge, criteria, and rules
2. **Open:** Community can submit attention patterns
3. **Voting:** Community votes on submissions (blind pairwise voting or ranked choice)
4. **Finalized:** Winners are calculated, prizes distributed
5. **Archived:** Challenge is preserved for reference

**Winner Determination:**

Two modes are supported:
- **Community Vote:** Winner is the submission with the highest vote score (upvotes − downvotes)
- **Expert Panel:** A designated panel of expert judges scores each submission on defined criteria

For community vote mode, the system uses a Bayesian ranking approach (similar to Reddit's "best" sorting) to avoid the "first‑mover advantage":
```
score = (upvotes / (upvotes + downvotes)) × log(upvotes + downvotes + 1)
```

### 28.3 Frontend Implementation

**Challenge Creation Wizard:**

A multi‑step form (like your existing experiment launcher) guides the creator through:
1. **Basic Info:** Title, description, category
2. **Criteria:** What makes a winning submission? (e.g., "Most interpretable attention pattern in BERT‑base Layer 5")
3. **Rules:** Submission limits, voting period, eligibility
4. **Prize Pool:** Platform credits, exclusive badges, featured spotlight
5. **Preview & Publish:** Review before making the challenge public

**Challenge Dashboard:**

Each challenge has a dedicated dashboard showing:
- **Leaderboard:** Ranked submissions with scores
- **Submit Button:** Opens a modal to select an existing attention exploration or create a new one
- **Voting Interface:** For active challenges, a voting panel for community members
- **Timeline:** Key dates (submission deadline, voting start, results announcement)

### 28.4 Integration Points

- **Leaderboards (existing):** Challenge rankings integrate with the global leaderboard system. The "Ladder" framework's approach is adopted: "a robust adaptive mechanism that updates leaderboards only when statistically significant improvements are detected."
- **Badge System (Zone 8):** Winners receive exclusive badges ("🏆 Challenge Champion").
- **Reputation (existing `gpu_reputation`):** Challenge participation and wins contribute to reputation score.
- **Portfolio:** Challenge achievements are prominently displayed.

---

## Feature 29: Inline Pattern Discussion Threads

### 29.1 Core Concept

Every attention pattern has threaded discussions anchored to specific elements: per‑head, per‑token‑pair, per‑layer. Supports @mentions, LaTeX equations, and embedded attention SVGs. Powered by the existing `community` Q&A infrastructure.

### 29.2 Backend Implementation

The implementation leverages the existing `community` app's `Question`, `Answer`, and `Comment` models, extended with polymorphic association to attention pattern elements.

**Database Extensions:**

A `DiscussionAnchor` model links a discussion thread to a specific element within an attention visualization:

The `element_type` field supports: `'head'`, `'token_pair'`, `'layer'`, `'entire_pattern'`. The `element_identifier` is a JSON field storing context‑specific data (e.g., `{ "layer": 5, "head": 7 }` for a head discussion, or `{ "query_index": 3, "key_index": 7 }` for a token‑pair discussion).

**WebSocket Integration for Real‑time Discussions:**

Discussions use Django Channels for real‑time updates. When a user posts a comment:
1. The comment is saved to the database via REST API
2. The backend broadcasts the comment to all users currently viewing the same pattern via WebSocket
3. Other users see the comment appear in real‑time without page refresh
4. The comment count badge updates instantly

### 29.3 Frontend Implementation

**Threaded Comment Component:**

The discussion panel renders below each attention pattern in Zone 2. The component uses a tree‑structured rendering for threaded replies (like Reddit or GitHub Issues).

**LaTeX Rendering:**

Mathematical expressions within comments are rendered using KaTeX (faster than MathJax, server‑side rendering compatible). Users wrap LaTeX in `$...$` for inline and `$$...$$` for block equations. The frontend uses `react‑katex` for client‑side rendering.

**Embedded Attention SVGs:**

Users can embed a snapshot of the current attention visualization in their comment by clicking a "Capture View" button. This generates a base64‑encoded SVG of the current canvas state and inserts it into the comment as an image. The SVG is stored as a `NoteImage` (using your existing `notes` infrastructure) and referenced by URL in the comment.

**@Mentions:**

The mention system is implemented using `tiptap` (a headless rich‑text editor) with the `@mentions` extension. As the user types `@`, a dropdown appears with suggested users (filtered from the platform user base). Selecting a user inserts a mention tag that triggers a notification.

### 29.4 Integration Points

- **Notifications (existing):** @mentions generate notifications via the existing notification pipeline.
- **Notes (existing `notes` app):** A "Save Discussion as Note" button exports the entire thread to a `ResearcherNote` for permanent reference.
- **Search (existing `search` infrastructure):** Discussion content is indexed for global search, enabling queries like "discussions about induction heads in BERT."

---

## Feature 30: One‑Click Share Cards

### 30.1 Core Concept

Export any attention visualization as a beautiful, social‑media‑optimized share card with one click. The card includes the heatmap, key statistics, model info, a QR code linking back to the interactive dashboard, and researcher attribution.

### 30.2 Implementation Architecture

**Server‑Side Rendering with Puppeteer:**

The share card is generated server‑side using Puppeteer (headless Chrome) to render a dedicated HTML template. This ensures:
- Consistent rendering across all devices (no CSS differences)
- High‑resolution output (2x for Retina displays)
- Inclusion of WebGL/Canvas content (which can't be captured by simple DOM serialization)

**Generation Pipeline:**

1. User clicks "Share" → Frontend sends `POST /api/attention/share-card/` with the current exploration state (model, layer, head, view type, zoom level)
2. Backend launches a Puppeteer instance (from a pool to avoid cold starts)
3. Puppeteer loads a dedicated share‑card HTML page with the same React components used in the dashboard
4. The page renders the attention visualization at the specified state
5. Puppeteer captures a screenshot of the card region
6. The image is saved to cloud storage (S3 or similar)
7. A unique short URL is generated
8. Response: `{ image_url, share_url, twitter_text, linkedin_text }`

**QR Code Generation:**

The QR code is generated server‑side using the `qrcode` Python library, encoding the full URL to the interactive dashboard with all parameters (model, layer, head, text). This ensures anyone scanning the code lands on the exact same visualization state.

### 30.3 Frontend Implementation

**Share Modal:**

Clicking the "Share" button opens a modal with multiple sharing options:
- **Copy Link:** Copies the short URL to clipboard
- **Download Image:** Downloads the PNG image
- **Twitter/X:** Pre‑filled tweet with image and text
- **LinkedIn:** Pre‑filled post
- **Reddit:** Pre‑filled post to r/MachineLearning
- **Embed Code:** Copy‑paste HTML snippet for embedding (see Feature 32)

**Preview:** The modal shows a live preview of the share card before sharing, with options to customize:
- Card title (default: pattern description)
- Subtitle (default: model + attention type)
- Show/hide specific statistics
- Color theme (dark/light)

---

## Feature 31: Attention Pattern Collections (Curated Galleries)

### 31.1 Core Concept

Users create themed collections of attention patterns. Collections can be public, team‑scoped, or private. Each collection has its own discussion thread and can be forked.

### 31.2 Backend Implementation

**Database Models:**

1. `AttentionCollection` — Collection metadata (name, description, creator, visibility, fork_of).
2. `CollectionItem` — Ordered membership of patterns in a collection, with optional notes.

**Visibility Model:**

Following Notion's teamspace model: `public` (visible to all, appears in gallery), `team` (visible to members of a specific organization), `private` (visible only to creator).

**Forking Collections:**

When a collection is forked, all items are copied into a new collection with a `forked_from` pointer. The forked collection can be independently modified. The lineage is tracked similarly to Feature 27.

### 31.3 Frontend Implementation

**Collection Grid View:**

The collections gallery page displays collections as cards in a responsive grid (1 column on mobile, 2 on tablet, 3 on desktop). Each card shows:
- Collection name and description
- Creator avatar and username
- Number of patterns in the collection
- Number of stars and forks
- Preview thumbnails of the first 4 patterns (mosaic layout)
- Visibility badge (public/team/private)

**Collection Detail Page:**

Opening a collection shows:
- Header with metadata and action buttons (Star, Fork, Share, Edit)
- Discussion thread below the header (powered by the same infrastructure as Feature 29)
- Pattern grid/list toggle with sorting options (date added, reaction count, pattern type)
- Each pattern card links to the full interactive visualization

**Curated Collections Leaderboard:**

The homepage and attention dashboard feature a "Trending Collections" section showing the most starred and forked collections this week. The ranking algorithm uses a time‑decay weighted score: `score = stars × 2 + forks × 3 / (hours_since_creation + 2) ^ 1.5`.

---

## Feature 32: Embed Anywhere Widget

### 32.1 Core Concept

Every attention pattern has an embeddable interactive widget (like YouTube embeds) that researchers can paste into personal websites, blogs, or paper supplementary materials.

### 32.2 Implementation Architecture

**iframe‑Based Embedding with postMessage Communication:**

The widget is implemented as a standalone React application that runs inside a sandboxed iframe. Communication between the host page and the widget uses the browser's postMessage API, validated by research: "The inter‑frame communication system uses the browser's postMessage API to enable secure, cross‑origin communication between the parent page (where the giscus script is embedded) and the iframe (where the widget renders). The protocol is bidirectional."

**Widget Application Architecture:**

Following the Chatwoot widget model: "The widget application is a standalone Vue.js 3 single‑page application that runs inside an iframe created by the SDK." In our implementation:
1. The widget is a React application built with Vite, outputting a single JavaScript bundle and HTML file
2. The iframe loads `https://neuralops.com/embed/{pattern_id}`
3. The host page includes a lightweight SDK script (`neuralops‑embed.js`) that creates the iframe and manages postMessage communication
4. The SDK supports responsive sizing: the iframe automatically resizes based on content height

**postMessage Protocol:**

The host page and widget communicate via a JSON‑RPC‑style protocol:

**Host → Widget:**
- `{ type: 'init', patternId: '...', theme: 'dark' | 'light', token: '...' }` — Initialize the widget
- `{ type: 'resize', width: 800, height: 600 }` — Resize the widget
- `{ type: 'setLayer', layer: 5 }` — Change the displayed layer
- `{ type: 'setHead', head: 7 }` — Change the displayed head

**Widget → Host:**
- `{ type: 'ready' }` — Widget is loaded and ready
- `{ type: 'resize', width: 800, height: 600 }` — Widget requests resize (for responsive content)
- `{ type: 'interaction', action: 'hover', data: {...} }` — User interacted with the visualization

**Security:**

The iframe is sandboxed with `sandbox="allow-scripts allow-same-origin"`. "The MCP App widget already runs inside a host‑sandboxed iframe, but HTML outputs need a nested iframe with sandbox='allow‑scripts' (no allow‑same‑origin) to prevent kernel‑generated scripts from accessing the parent's PostMessageTransport." The widget validates the origin of postMessage events to prevent cross‑origin attacks.

### 32.3 Analytics

The widget SDK includes analytics tracking (using the existing platform analytics infrastructure):
- **Impressions:** How many times the widget was loaded
- **Interactions:** Hover, click, zoom events within the widget
- **Session duration:** How long users interact with the embedded visualization
- **Referrer:** Which websites are embedding the widget

These analytics are displayed on the pattern owner's portfolio dashboard.

---

## Feature 33: Attention Portfolio — Researcher Showcase

### 33.1 Core Concept

Every researcher has a public portfolio showing: published patterns, collections curated, challenges won, reputation score, activity graph, expertise tags. The portfolio is embeddable on personal websites and serves as a professional credential.

### 33.2 Backend Implementation

**Data Aggregation Service:**

The portfolio is powered by an aggregation service that collects data from across the platform:
- **Patterns Published:** From `AttentionExploration` model
- **Collections Curated:** From `AttentionCollection` model
- **Challenges Won:** From `ChallengeSubmission` + `AttentionChallenge` models
- **Reactions Received:** Sum across all published patterns
- **Reputation Score:** From `gpu_reputation` app
- **Badges Earned:** From `UserBadge` model (Zone 8)
- **Activity History:** From `ActivityLog` model (existing)

**API Endpoint:** `GET /api/attention/portfolio/{username}/`

Returns a comprehensive JSON object with all portfolio data. The endpoint is optimized with Redis caching (5‑minute TTL for public portfolios, real‑time for the owner's view).

### 33.3 Frontend Implementation

**Portfolio Layout:**

The portfolio page follows a layout inspired by GitHub profile pages and Kaggle profiles:

**Header Section:**
- Avatar, display name, username
- Headline/bio from user profile
- Reputation score with tier badge
- Social links (GitHub, Twitter, LinkedIn, Google Scholar)
- "Edit Profile" button (for the owner)

**Stats Bar:**
- Total patterns published
- Total collections curated
- Total challenges won
- Total reactions received
- Total forks of their patterns
- Followers count

**Activity Graph (GitHub‑Style Contribution Chart):**

A heatmap showing attention‑related activity over the past year. Each cell represents a day, colored by activity level (number of patterns published, reactions given, discussions participated in). The implementation uses D3.js with a calendar heatmap layout.

**Tabs Section:**

1. **Patterns:** Grid of published attention patterns with thumbnail previews, reaction counts, and fork counts
2. **Collections:** Grid of curated collections
3. **Challenges:** List of challenges created and participated in
4. **Badges:** Grid of earned badges with tooltips
5. **Activity:** Chronological feed of recent actions

**Expertise Tags:**

Auto‑generated tags based on the researcher's activity patterns:
- "Induction Head Expert" (if they've published 5+ patterns classified as induction heads)
- "Vision Attention Specialist" (if they've analyzed 10+ vision transformer patterns)
- "Compression Researcher" (if they've used the compression pipeline on attention heads)
- "Community Mentor" (if they've answered 20+ discussion questions)

### 33.4 Integration Points

- **Embed Widget (Feature 32):** The entire portfolio can be embedded on external websites using the same widget infrastructure.
- **Profile (existing `profiles` app):** The Attention Portfolio extends the existing user profile with attention‑specific sections.
- **Social Graph (existing `social_graph`):** Follow/unfollow buttons on the portfolio, follower/following counts.
- **Messaging (existing `messaging` app):** A "Message" button on the portfolio opens a direct message with the researcher.

---

## Cross‑Cutting Concerns

### Security & Access Control

- **Row‑Level Security:** All social features respect the existing platform permission model. Patterns marked as private are only visible to their creator and explicitly shared users.
- **Rate Limiting:** Reaction and vote endpoints are rate‑limited to 60 requests per minute per user (via Django REST Framework throttling).
- **Content Moderation:** Discussion comments are filtered through a profanity filter and optional AI‑based toxicity detection before publication.
- **GDPR Compliance:** Users can export all their social data (reactions, comments, votes) via a "Download My Data" button, and request deletion of their social footprint.

### Performance Optimization

- **CDN Caching:** Share card images and widget assets are served via CDN with long cache TTLs (1 year for versioned assets).
- **Lazy Loading:** The discussion thread component is lazy‑loaded only when the user scrolls to the discussion section.
- **Virtual Scrolling:** Collections with hundreds of patterns use virtual scrolling (via `@tanstack/react‑virtual`) to maintain performance.
- **WebSocket Connection Pooling:** Multiple Zone 7 features share a single WebSocket connection, multiplexed by message type.

### Testing Strategy

| Test Type | Scope | Tools |
|-----------|-------|-------|
| Unit Tests | Individual reaction/vote/discussion components | Vitest + React Testing Library |
| Integration Tests | End‑to‑end vote flow, fork lineage chain | Playwright |
| Load Tests | Arena voting under 1000 concurrent users | Locust |
| Accessibility | WCAG 2.1 AA compliance for all interactive elements | axe‑core + pa11y |
| Visual Regression | Share card rendering consistency | Percy (for server‑side rendered cards) |

---

## 📊 Implementation Priority Matrix

| Feature | Complexity | Impact | Dependencies | Estimated Effort |
|---------|-----------|--------|-------------|-----------------|
| Feature 25: Reaction System | Medium | High | Event bus, Redis | 2 weeks |
| Feature 26: Attention Arena | High | Very High | Elo engine, session mgmt | 3 weeks |
| Feature 27: Fork & Remix | Medium | High | Lineage DB, graph viz | 2 weeks |
| Feature 28: Challenge Creator | High | High | Leaderboard integration | 3 weeks |
| Feature 29: Inline Discussions | Medium | Medium | Community Q&A, WebSocket | 2 weeks |
| Feature 30: Share Cards | Medium | Medium | Puppeteer, CDN | 2 weeks |
| Feature 31: Collections | Medium | Medium | Fork system | 2 weeks |
| Feature 32: Embed Widget | High | Medium | iframe SDK, postMessage | 3 weeks |
| Feature 33: Portfolio | Medium | High | Aggregation service | 2 weeks |

---

## 🔗 Integration Map

```
Zone 7: Social & Community Layer
│
├── Reaction System ──────────────→ Recommendation Engine
│                                   → Leaderboards (reaction leaderboards)
│                                   → Portfolios (reaction counts)
│                                   → Activity Feed
│
├── Attention Arena ──────────────→ Elo Rating Engine
│                                   → Leaderboards (Elo rankings)
│                                   → Reputation (vote consistency scoring)
│                                   → Dataset Export (preference data)
│
├── Fork & Remix ────────────────→ Lineage Graph (D3 force‑directed)
│                                   → Activity Feed
│                                   → Portfolios (fork counts)
│
├── Challenge Creator ────────────→ Leaderboards (challenge rankings)
│                                   → Badge System (challenge badges)
│                                   → Reputation (participation scoring)
│
├── Inline Discussions ───────────→ Community Q&A (reuses models)
│                                   → Notifications (@mentions)
│                                   → Notes (export discussion as note)
│                                   → WebSocket (real‑time updates)
│
├── Share Cards ─────────────────→ Puppeteer rendering service
│                                   → CDN (card image hosting)
│                                   → Widget SDK (embed preview)
│
├── Collections ─────────────────→ Fork system (collection forking)
│                                   → Discussion threads
│                                   → Search indexing
│
├── Embed Widget ────────────────→ iframe SDK + postMessage
│                                   → Analytics (impressions, interactions)
│                                   → CDN (widget assets)
│
└── Portfolio ───────────────────→ Profiles (extends existing)
                                    → Social Graph (follow/unfollow)
                                    → Messaging (direct message)
                                    → Embed Widget (portfolio embed)
```

All nine features are designed to work with your existing React 19 + TypeScript + Tailwind CSS frontend, Django REST Framework + FastAPI backend, Django Channels WebSocket infrastructure, and the full suite of platform apps (community, leaderboards, gpu_reputation, notes, portfolios, profiles, messaging, search, activity). Each feature is independently deployable, enabling incremental rollout based on priority.