# 🎮 Zone 8: Gamification & Progress — Complete Implementation Specification

## Overview

Zone 8 is the **motivational engine** of the entire Attention Visualization Dashboard. It sits as a **persistent sidebar panel** (collapsible, 320px wide on desktop) on the right edge of the dashboard, always visible regardless of which zone the user is interacting with. It serves five tightly integrated functions: XP tracking and tier progression, daily habit‑forming puzzles, a multi‑category badge system, real‑time leaderboards, and a bounty marketplace. Each function connects bidirectionally to every other zone: completing a visualization exploration (Zone 2) awards XP (Zone 8); reaching a new tier (Zone 8) unlocks premium attention types (Zone 4); winning a challenge (Zone 8) auto‑generates a researcher note (Zone 7).

The Zone 8 panel uses **motion‑rich micro‑interactions** (Framer Motion spring animations, particle burst effects on level‑up, Lottie badge animations) and follows the same glass‑morphism design language (`glass`, `bg-white/5`, `border-white/10`) as the rest of the dashboard.

---

## Feature 34: XP & Progression Tiers — The Core Scoring Engine

### 34.1 Architecture & Technology Stack

The XP engine follows the **event‑driven, bounded‑context architecture** established by the Hephaestus gamification engine, which splits gamification into distinct bounded contexts: the **Activity Module** (XP tracking) and the **Achievement Module** (badge/level resolution). This separation allows the XP engine to scale independently of badge logic.

**Core technologies:**

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Event Bus** | Apache Kafka (self‑hosted via KRaft, 3‑broker minimum) or Redis Streams (lighter, for <100K events/sec) | Decouples action producers (Zone 2 clicks, Zone 3 formula exploration) from XP consumers. Each action emits an event; the XP engine consumes and scores. |
| **XP Computation** | Custom Spring Boot microservice (Java 21, virtual threads) or FastAPI (Python 3.12, async) | Java virtual threads provide massive concurrency for real‑time XP calculation; Python FastAPI enables tight integration with ML infrastructure. |
| **State Store** | PostgreSQL 16 with TimescaleDB extension (for time‑series XP history) + Redis Stack (RedisJSON + RediSearch for real‑time level cache) | TimescaleDB handles the write‑heavy XP event log and enables time‑range queries for "XP earned this week." Redis holds a per‑user JSON document `user:{id}:xp` with `{ currentXP, currentLevel, tierName, nextLevelXP }` for O(1) reads. |
| **Notification** | Server‑Sent Events (SSE) over HTTP/2 or WebSocket (existing Channels infrastructure) | Level‑up events are pushed in real time. SSE is preferred for unidirectional notifications; WebSocket is used if bidirectional chat/discussion is needed. |
| **Rule Engine** | Netflix Conductor (workflow orchestration) or Drools (Drools 9.x, rule‑based) | Different actions award different XP amounts. A lightweight rule engine allows non‑engineers to modify XP tables without redeployment. See Feature 36 for rule engine architecture. |

### 34.2 XP Award Table (Configurable via YAML + Admin UI)

The XP award values follow a **logarithmic‑decay model**: common actions (viewing a pattern) award small XP; rare, high‑effort actions (winning a challenge, publishing a paper) award large XP. This prevents XP inflation and encourages depth over breadth.

```yaml
# xp_rules.yaml — deployed to the rule engine, editable via admin panel
xp_awards:
  # ── Zone 2 (Visualization) ──
  view_pattern: 5                    # per unique pattern viewed
  change_layer: 1                    # per layer switch (capped at 20/day)
  change_head: 1                     # per head switch (capped at 20/day)
  use_3d_view: 10                    # first 3D view per session
  use_sankey_view: 10
  zoom_to_token: 3                   # clicking a specific token pair
  export_svg: 5                      # exporting a visualization

  # ── Zone 3 (Math Panel) ──
  view_equation_step: 2              # scrubbing through math animation
  export_numerical_data: 8           # exporting CSV of attention values

  # ── Zone 5 (Analytics) ──
  run_ablation: 15                   # toggling a head ablation
  view_logit_lens: 10
  view_semantic_pathway: 12

  # ── Zone 6 (Interpretability) ──
  run_causal_trace: 25               # knockout experiment
  annotate_pattern: 15               # adding a linguistic annotation
  publish_circuit_trace: 30          # publishing a QK/OV circuit graph

  # ── Zone 7 (Social) ──
  publish_visualization: 50          # creating a public attention post
  receive_star: 10                   # someone stars your pattern
  give_star: 2                       # starring someone else's pattern
  write_comment: 15                  # posting a discussion comment
  receive_comment_reply: 5           # someone replies to your comment
  fork_pattern: 20                   # forking another researcher's pattern
  win_arena_vote: 8                  # your pattern wins a blind comparison
  win_challenge: 200                 # first place in a community challenge
  curate_collection: 40              # creating a curated collection
  share_on_social: 5                 # sharing a pattern on Twitter/LinkedIn

  # ── Zone 4 (Training) ──
  launch_experiment: 100             # starting a training run from attention viz
  complete_experiment: 300           # training run successfully completes
  publish_model: 500                 # publishing a compressed/trained model

  # ── Streak ──
  daily_puzzle_correct: 25           # correct daily puzzle answer
  daily_puzzle_streak_bonus: 5       # extra per consecutive day (caps at 50)

  # ── Negative (rare) ──
  pattern_flagged_and_removed: -50   # community moderation removal
  arena_vote_reversed: -10           # vote found to be fraudulent
```

### 34.3 Level Progression Table (Logarithmic Scaling)

The level curve follows **Dreyfus model progression**: early levels are rapid (rewarding new users), later levels require exponentially more XP. This is implemented as a **piecewise exponential function**:

```
XP_required(level) = baseXP × (growthFactor ^ level)

Level 1 (Novice Explorer):      0 XP
Level 2 (Pattern Scout):       150 XP    (growthFactor = 1.5)
Level 3:                       375 XP
Level 4:                       750 XP
Level 5 (Head Hunter):        1,500 XP   (growthFactor increases to 1.8)
Level 6:                      2,700 XP
Level 7:                      4,860 XP
Level 8:                      8,748 XP
Level 9:                     15,746 XP
Level 10 (Attention Sage):   28,344 XP
Level 11:                    51,019 XP
Level 12:                    91,834 XP
Level 13:                   165,301 XP
Level 14:                   297,542 XP
Level 15 (Grandmaster):     535,575 XP
```

The mathematical model for this curve is inspired by the **Italearn XP system**, which uses "XP thresholds with increasing gaps (Level 1 = 0, Level 2 = 100, Level 3 = 250, etc.)" and the **TRACE‑Weighted XP** approach from NeuroLearn, which "rewards depth over throughput" by weighting XP based on the epistemic depth of the action.

### 34.4 Level‑Up Event Architecture

```
User Action → Kafka Topic "user.actions" → XP Calculation Service → 
  → Redis INCRBY user:{id}:xp → Check Level Threshold → 
  → If threshold exceeded: 
      → PostgreSQL UPDATE user_level → 
      → Redis publish "levelup:{userId}" → 
      → WebSocket broadcast to user's session →
      → Frontend triggers: particle burst animation + 
        Lottie badge animation + 
        toast notification + 
        confetti effect (canvas-confetti library)
```

**Level‑up animation stack:**
- **canvas‑confetti** (`npm: canvas-confetti`): Fires a 2‑second confetti burst from the center of the screen with custom colors matching the new tier (bronze→silver→gold→platinum→diamond).
- **Framer Motion**: The Zone 8 panel slides out slightly, the level badge scales from 0→1.2→1.0 (spring animation, stiffness=300, damping=20).
- **Lottie**: A pre‑rendered Lottie animation (JSON, ~50KB) plays a tier‑specific celebration (e.g., a torch lighting for Novice→Scout, a crown for Sage→Grandmaster).
- **Sound**: A subtle chime (`howler.js`, ~15KB audio sprite) plays on level‑up (can be disabled).

### 34.5 Database Schema

```sql
-- PostgreSQL + TimescaleDB hypertable for XP event log
CREATE TABLE xp_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    action_type VARCHAR(50) NOT NULL,        -- e.g., 'view_pattern', 'publish_visualization'
    xp_amount INTEGER NOT NULL,
    source_zone VARCHAR(20),                 -- 'zone2', 'zone3', etc.
    source_entity_id UUID,                   -- the pattern ID, comment ID, etc.
    metadata JSONB DEFAULT '{}',             -- additional context
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series queries
SELECT create_hypertable('xp_events', 'created_at');

-- User level state (in PostgreSQL, mirrored in Redis)
CREATE TABLE user_gamification_state (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    current_xp INTEGER DEFAULT 0,
    current_level INTEGER DEFAULT 1,
    current_tier VARCHAR(30) DEFAULT 'Novice Explorer',
    total_xp_earned INTEGER DEFAULT 0,
    streak_count INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    last_activity_date DATE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for leaderboard queries
CREATE INDEX idx_user_xp ON user_gamification_state(current_xp DESC);
CREATE INDEX idx_user_level ON user_gamification_state(current_level DESC);
```

**Redis cache structure:**
```json
// Key: user:{id}:xp
{
  "currentXP": 2847,
  "currentLevel": 5,
  "currentTier": "Head Hunter",
  "nextLevelXP": 4860,
  "xpToNextLevel": 2013,
  "progressPercent": 58.6,
  "streakCount": 12,
  "lastActivityDate": "2026-05-26"
}
```

### 34.6 Anti‑Abuse Mechanisms

- **Rate limiting per action type**: Using Redis token bucket (`redis-cell` or custom Lua script). Example: `view_pattern` capped at 50 XP/day from viewing (50 views × 5 XP = 250 XP/day max from viewing). Beyond that, views still count but award 0 XP.
- **Duplicate detection**: XP is awarded only once per unique `(user_id, action_type, source_entity_id)` combination. The `xp_events` table has a unique constraint on these three columns.
- **Suspicious pattern detection**: A background job (Celery, every 15 minutes) runs anomaly detection (DBSCAN clustering on XP/hour rate). Users with XP/hour > 3σ above the mean are flagged for review.
- **Bot detection**: Integration with reCAPTCHA v3 (or hCaptcha) for account creation. All XP‑awarding API endpoints require a valid CSRF token + reCAPTCHA score > 0.5.

---

## Feature 35: Daily Attention Puzzle — Habit‑Forming Mini‑Challenge

### 35.1 Architecture & Technology Stack

The Daily Puzzle system follows the **Wordle‑style habit loop** architecture: one puzzle per day, released at midnight UTC, with a 24‑hour window to solve. Users can play any previous day's puzzle (for practice, no XP), but only today's puzzle counts toward their streak.

**Core technologies:**

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Puzzle Generation** | Custom Python service (FastAPI) using OpenAI GPT‑4o API or Anthropic Claude for puzzle creation + a fallback deterministic generator for offline resilience | Puzzles are generated nightly (cron job, 00:00 UTC) using an LLM with a structured output schema (JSON). A deterministic backup generator (rule‑based) ensures puzzles are always available even if the LLM API is down. |
| **Puzzle Storage** | PostgreSQL + Redis Cache | Each day's puzzle is stored in PostgreSQL (`daily_puzzles` table). Today's puzzle is cached in Redis with TTL=24h for O(1) reads. |
| **State Management** | Redis `user:{id}:puzzle:{date}` JSON document | Stores `{ played: true/false, answer: "induction_head", correct: true/false, timeTaken: 45 }`. |
| **Streak Calculation** | PostgreSQL window function or Redis bitmap | A Redis bitmap `streak:user:{id}` tracks play days (bit=1 if played). The streak is computed as the number of consecutive 1s counting backward from today. |

### 35.2 Puzzle Types (7‑Day Rotation)

Each day of the week features a different puzzle type, ensuring variety and testing different attention‑related skills:

| Day | Puzzle Type | Description | Difficulty |
|-----|------------|-------------|------------|
| **Monday** | "Guess the Head Type" | An attention heatmap is shown without labels. The user must identify whether the head is positional, semantic, delimiter, induction, or redundant. Multiple choice (5 options). | Easy (50% baseline guess rate) |
| **Tuesday** | "Spot the Difference" | Two attention heatmaps are shown side by side (same model, same text, but one head ablated). The user must identify which 3 token pairs show the biggest attention difference. Click‑to‑select on the heatmap. | Medium |
| **Wednesday** | "Match the Model" | An attention pattern is shown. The user must identify which model architecture (BERT‑base, GPT‑2, RoBERTa, T5, Llama‑2‑7B) produced it. Multiple choice. | Hard |
| **Thursday** | "Token Detective" | Given an attention weight distribution (a row of the attention matrix, all values shown), the user must infer what the input text might be. They type a free‑text guess; the system scores based on semantic similarity (sentence‑transformer cosine similarity > 0.7 = correct). | Hard |
| **Friday** | "Rank the Heads" | Five heads from the same layer are shown. The user must rank them by importance (1 = most important, 5 = least). Correct ordering awards bonus XP. | Medium |
| **Saturday** | "Circuit Challenge" | A QK circuit is partially shown. The user must complete the missing edge weights to make the circuit produce the shown attention pattern. Drag‑and‑drop weights. | Hard |
| **Sunday** | "Free Play Showcase" | No puzzle. Instead, users submit their most creative attention visualization from the past week. Community votes determine the winner. Winner gets 100 bonus XP. | Community |

### 35.3 Puzzle Answer Verification Flow

```
1. User opens dashboard → GET /api/puzzles/today → 
   Response: { puzzleId, puzzleType, puzzleData, puzzleDate }

2. User plays → Frontend tracks time spent, interactions

3. User submits → POST /api/puzzles/submit { puzzleId, answer, timeTaken }
   → Backend validates:
      a. puzzleDate == today (in UTC)
      b. Not already answered (check Redis user:{id}:puzzle:{date})
      c. Anti‑cheat: timeTaken > 5 seconds (not a bot instant‑submit)
   
4. If correct:
   → Redis SET user:{id}:puzzle:{date} { correct: true, timeTaken }
   → Redis INCRBY user:{id}:xp by daily_puzzle_correct (25 XP)
   → Update streak (Redis BITFIELD streak:user:{id} SET u1 #{dayOfYear} 1)
   → Return: { correct: true, xpAwarded: 25, streakNow: 8, explanation: "..." }
   
5. If incorrect:
   → Redis SET user:{id}:puzzle:{date} { correct: false, timeTaken }
   → Break streak (Redis BITFIELD streak:user:{id} SET u1 #{dayOfYear} 0)
   → Return: { correct: false, explanation: "This was a positional head because..." }
```

### 35.4 Streak Visualization

The streak is displayed in Zone 8 as a **7‑day mini‑calendar** (inspired by Duolingo's streak flame and GitHub's contribution graph):

```
┌──────────────────────────────────────────┐
│ 🔥 12‑Day Streak!                        │
│                                          │
│  M  T  W  T  F  S  S                     │
│  ✅ ✅ ✅ ✅ ✅ ✅ ✅  ← This week          │
│  ✅ ✅ ✅ ✅ ✅ ⬜ ⬜  ← Last week          │
│                                          │
│  Longest: 42 days  ·  Total: 156 puzzles │
└──────────────────────────────────────────┘
```

- The flame icon (🔥) animates (CSS `@keyframes pulse`) when the streak reaches multiples of 7 (7, 14, 21, …).
- At 7 days: a bronze flame; at 30 days: silver; at 100 days: gold; at 365 days: diamond flame.
- Missing a day resets the streak to 0 (the flame extinguishes with a CSS `fadeOut` animation).
- A **"Streak Freeze"** power‑up (earned at level 5, one per month) allows missing one day without breaking the streak.

### 35.5 Anti‑Cheat for Puzzles

- **Server‑side answer validation only**: The correct answer is never sent to the client until after submission. The `puzzleData` field does not contain the answer.
- **Time‑based minimum**: Answers submitted in <5 seconds are rejected (impossible for a human to analyze an attention heatmap).
- **IP‑based rate limiting**: Max 3 submissions per puzzle per IP address (handles network retries).
- **Pattern randomization**: For multiple‑choice puzzles, the option order is randomized per user to prevent answer sharing.
- **CAPTCHA on suspicious activity**: If a user submits correct answers in <10 seconds for 3 consecutive days, a CAPTCHA is triggered.

---

## Feature 36: Badge System — Rule Engine Architecture

### 36.1 Architecture & Technology Stack

The badge system follows the **DoorDash Badge Serving Framework (BSF)** architecture, which is a "configuration‑driven platform that makes it easy for developers to create, manage, and deliver badges for products across the ecosystem." The key insight is that badge logic is **externalized from application code** into a YAML‑based rule engine, allowing new badges to be added without redeployment.

**Core technologies:**

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Rule Engine** | RuleGo (Go‑based, lightweight, embedded, orchestratable component‑based rule engine) or Drools 9.x (Java, more mature, KIE Server for REST API) | RuleGo is "a lightweight, high‑performance, embedded, orchestrable component‑based rule engine built on Go" that can "help you quickly build loosely coupled and flexible systems." Drools provides "point‑and‑click badge criteria, tiers, expirations, streaks, and multi‑step achievements." |
| **Rule Storage** | Git‑backed YAML files (monorepo) + PostgreSQL `badge_definitions` table (for runtime caching) | Badge rules are defined in YAML, version‑controlled in Git. On deploy, a CI/CD pipeline validates the rules, then loads them into PostgreSQL. The badge engine reads from PostgreSQL at startup and caches in memory. |
| **Event Bridge** | Kafka (same cluster as XP engine) + Debezium CDC (change data capture from PostgreSQL) | When an XP event is written, Debezium captures it and pushes to Kafka. The badge engine consumes these events, evaluates all badge rules, and awards badges in real time. |
| **Badge Storage** | PostgreSQL `user_badges` table + Redis Set `user:{id}:badges` for O(1) membership check | Badges are persisted in PostgreSQL. Redis holds a Set of badge IDs per user for fast UI rendering. |
| **Notification** | WebSocket (Channels) for real‑time award toast + Email (SendGrid/Mailgun) for weekly badge digest | Badge awards trigger an immediate WebSocket notification and are included in the weekly "Your Progress" email. |

### 36.2 Badge Rule Definition Language (YAML‑Based)

Each badge is defined as a YAML file. The rule engine parses this YAML at startup and evaluates it against every incoming XP event.

```yaml
# badges/head_hunter_gold.yml
badge:
  id: "head_hunter_gold"
  name: "Head Hunter (Gold)"
  description: "Identified 50 distinct attention head types across all models"
  category: "discovery"
  tier: "gold"                    # bronze | silver | gold | platinum
  icon: "🔬"
  lottie_animation: "head_hunter_gold.json"   # Lottie file for award animation

  # Conditions — ALL must be true (AND logic)
  conditions:
    - field: "user_stats.distinct_head_types_identified"
      operator: "gte"
      value: 50

  # OR conditions — ANY can be true
  or_conditions: []

  # Time‑based conditions
  validity:
    type: "permanent"              # permanent | expires_after_days | seasonal
    # expires_after_days: 365     # only if type = expires_after_days

  # Award notification template
  notification:
    title: "🏆 Badge Unlocked!"
    body: "You've identified 50 distinct attention head types. You're a true Head Hunter!"
    cta_url: "/dashboard/attention-lab?badge=head_hunter_gold"
```

```yaml
# badges/streak_master_platinum.yml
badge:
  id: "streak_master_platinum"
  name: "Streak Master (Platinum)"
  description: "Maintained a 365‑day daily puzzle streak"
  category: "engagement"
  tier: "platinum"
  icon: "🔥"

  conditions:
    - field: "user_stats.longest_streak"
      operator: "gte"
      value: 365

  notification:
    title: "💎 STREAK MASTER!"
    body: "365 days of daily puzzles. You are among the top 0.1% of researchers."
    cta_url: "/dashboard/profile"
```

```yaml
# badges/community_curator_silver.yml
badge:
  id: "community_curator_silver"
  name: "Curator (Silver)"
  description: "Created a public collection with 100+ followers"

  conditions:
    - field: "user_stats.max_collection_followers"
      operator: "gte"
      value: 100

  or_conditions:
    - field: "user_stats.collections_curated"
      operator: "gte"
      value: 10
      note: "OR, if you've created 10+ smaller collections"
```

### 36.3 Badge Evaluation Pipeline (Event‑Driven)

```
Kafka Topic "user.actions" → Badge Evaluation Service (RuleGo/Drools) →
  → For each event:
      → Load all badge rules from cache (in‑memory, refreshed every 5 min)
      → Filter rules: only rules whose conditions reference the event's action_type
      → For each filtered rule:
          → Query current user stats from Redis user:{id}:stats (JSON document)
          → Evaluate conditions (RuleGo expression engine or Drools MVEL)
          → If all conditions met:
              → Check Redis Set user:{id}:badges (avoid duplicate awards)
              → If not already awarded:
                  → PostgreSQL INSERT INTO user_badges
                  → Redis SADD user:{id}:badges {badgeId}
                  → Redis PUBLISH badge:awarded {userId, badgeId, badgeName}
                  → WebSocket push to user session
                  → (Optional) Email notification (SendGrid template)
```

### 36.4 Badge Award UI (Frontend)

When a badge is awarded, the frontend receives a WebSocket message:

```json
{
  "type": "badge_awarded",
  "badgeId": "head_hunter_gold",
  "badgeName": "Head Hunter (Gold)",
  "badgeIcon": "🔬",
  "badgeTier": "gold",
  "lottieAnimation": "https://cdn.neuralops.com/badges/head_hunter_gold.json",
  "awardedAt": "2026-05-26T14:32:00Z"
}
```

The frontend then:

1. **Pauses** all ongoing animations (via Framer Motion `AnimatePresence`).
2. **Darkens** the background (semi‑transparent overlay, `backdrop-filter: blur(8px)`).
3. **Plays** the Lottie animation at center screen (3 seconds, scales from 0→1.5→1.0).
4. **Shows** the badge name and description with a typewriter effect (`typed.js` or CSS animation).
5. **Fires** confetti (`canvas-confetti`, gold particles for gold tier, silver for silver, etc.).
6. **Adds** the badge to the Zone 8 badge grid with a `layoutId` animation (Framer Motion).

### 36.5 Badge Categories (25+ Badges)

| Category | Bronze (10) | Silver (8) | Gold (5) | Platinum (2) |
|----------|------------|------------|----------|--------------|
| **Discovery** | First Pattern Viewed, 10 Head Types, Induction Head Finder, Positional Head Spotter, Semantic Head Spotter, Delimiter Head Spotter, 100 Patterns Viewed, Cross‑Model Explorer, QK Circuit Viewer, OV Circuit Viewer | 50 Head Types, 500 Patterns, Ablation Expert, Causal Trace Expert, Logit Lens Expert, Pattern Evolution Watcher, Cross‑Architecture Explorer, Multi‑Model Comparer | 200 Head Types, 2000 Patterns, Published Discovery (peer‑reviewed), Circuit Tracer, AttentionViz Master | 1000 Head Types, Grand Discovery (community‑verified novel pattern) |
| **Community** | First Comment, 10 Comments, First Star Given, First Star Received, 10 Stars Received, Pattern Published, Pattern Forked (by someone else), Joined Circle, Voted in Arena, Profile Complete | 50 Comments, 100 Stars Received, 10 Patterns Published, 5 Collections Curated, Circle Creator, Challenge Participant, 100 Arena Votes, Mentorship (paired with novice) | 500 Stars Received, 50 Patterns Published, Challenge Winner, Top 10 Curator, Featured Spotlight | 5000 Stars Received, Community Legend (lifetime achievement) |
| **Engagement** | 3‑Day Streak, 7‑Day Streak, First Daily Puzzle, First Challenge, First Bounty, All Views Explored, Profile Shared, Dashboard Customized, Export Pattern, First Paper Section | 30‑Day Streak, 100‑Day Streak, 50 Puzzles, 10 Challenges, 5 Bounties, Widget Embedded, API Key Created, Multi‑Session Workshop | 365‑Day Streak, 500 Puzzles, 50 Challenges, 20 Bounties, Published Research Paper | 1000‑Day Streak |
| **Expertise** | Positional Expert (100 positional heads identified), Semantic Expert, Induction Expert, Delimiter Expert, Compression Expert (10 patterns from compressed models), Galerkin Expert, TLinFormer Expert, PADRe Expert, Spectraformer Expert, RPC Expert | Cross‑Architecture Expert (all 6 attention types), Speed Demon (fastest puzzle solver, top 5%), Circuit Master (50 circuits published), Annotation Master (100 annotations), Review Master (50 peer reviews), Challenge Judge, Bounty Hunter (10 bounties completed), Accuracy Master (95% arena vote accuracy) | Published Author (paper with embedded patterns), Top 1% Contributor, 500K XP, 100 Badges Total | 1M XP, All Categories Mastered |

---

## Feature 37: Leaderboards — Real‑Time, Multi‑Dimensional

### 37.1 Architecture & Technology Stack

The leaderboard system uses **Redis Sorted Sets** as the primary data structure, which is "very efficient since data is being managed in‑memory, allowing for fast read and write operations." The architecture follows the **multi‑shard, pub/sub broadcast** pattern recommended for production systems:

**Core technologies:**

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Primary Store** | Redis Sorted Sets (ZSET) | O(log N) insertion, O(log N + M) range queries for top‑M. Each leaderboard dimension has its own ZSET key: `lb:xp:weekly`, `lb:xp:monthly`, `lb:xp:alltime`, `lb:patterns:weekly`, etc. |
| **Periodic Archival** | Redis → PostgreSQL nightly sync (Celery task) + ClickHouse for analytics | Every night at 00:00 UTC, the weekly leaderboard is archived to PostgreSQL and a new weekly ZSET is created. ClickHouse (columnar DB) stores historical leaderboard snapshots for trend analysis. |
| **Real‑Time Broadcast** | Centrifugo (WebSocket‑native, Redis‑backed) or Django Channels + Redis Pub/Sub | When a user's score changes, a message is published to `leaderboard:updates`. All connected clients subscribed to that leaderboard receive the update. Centrifugo is recommended for production as it handles "realtime progress, session events, and broadcasts using Redis Pub/Sub + Socket.IO." |
| **Top‑N Caching** | Redis String (JSON) `lb:xp:weekly:top100` | The top 100 entries are cached as a JSON string and updated every 5 seconds (or on each write, if write volume is low). This avoids recomputing ZSET ranges on every read. |
| **Sharding** | 128‑shard ZSET for "all‑time XP" leaderboard (for scale > 10M users) | Following the pattern: "If I need the server‑wide Top 10, I must take the Top 10 from each of the 128 ZSETs and merge‑sort them in application memory." This avoids a single massive ZSET that exceeds Redis memory. |

### 37.2 Leaderboard Dimensions

| Leaderboard | Redis Key Pattern | Scoring Metric | Period |
|-------------|-------------------|----------------|--------|
| **Top XP Earners** | `lb:xp:{period}` | Total XP earned in period | Weekly, Monthly, All‑Time |
| **Top Pattern Discoverers** | `lb:patterns:{period}` | Number of head types identified | Weekly, Monthly, All‑Time |
| **Top Publishers** | `lb:publishers:{period}` | Number of attention visualizations published | Weekly, Monthly, All‑Time |
| **Top Curators** | `lb:curators:{period}` | Total followers across collections | Weekly, Monthly, All‑Time |
| **Top Reviewers** | `lb:reviewers:{period}` | Number of peer reviews completed | Weekly, Monthly, All‑Time |
| **Arena Elo Ratings** | `lb:arena:elo` | Elo rating from blind pairwise voting | All‑Time (updates continuously) |
| **Challenge Champions** | `lb:challenges:{period}` | Challenge wins | Weekly, Monthly, All‑Time |
| **Streak Leaders** | `lb:streaks:current` | Current active streak length | Current |
| **Bounty Hunters** | `lb:bounties:{period}` | Total bounty rewards earned | Weekly, Monthly, All‑Time |

### 37.3 Leaderboard UI Design

The leaderboard panel in Zone 8 displays a **tabbed interface** with three tabs: "This Week," "This Month," and "All‑Time." Each tab shows a scrollable list of the top 100 entries.

```
┌──────────────────────────────────────────┐
│ 🏆 Leaderboard                           │
│ [This Week] [This Month] [All‑Time]      │
├──────────────────────────────────────────┤
│                                          │
│  #1  👤 Alice Chen                       │
│      🥇 Grandmaster · 12,450 XP          │
│      ────────────────────────────────    │
│  #2  👤 Bob Kumar                        │
│      🥇 Grandmaster · 11,230 XP          │
│      ────────────────────────────────    │
│  #3  👤 Carol Zhang                      │
│      🥈 Attention Sage · 9,870 XP        │
│      ────────────────────────────────    │
│  ...                                     │
│                                          │
│  #42 👤 You (current user)               │
│      🥉 Head Hunter · 2,847 XP           │
│      ↑ 3 positions since yesterday       │
│                                          │
└──────────────────────────────────────────┘
```

**Key UI features:**
- The current user's row is **highlighted** with a `bg-neon-blue/10` background and a left border accent.
- The user's row shows **position change** (↑ 3, ↓ 1, — 0) since yesterday.
- **Avatars** use the user's profile picture from the `profiles` app.
- **Tiers** are displayed as colored badges (🥇 Grandmaster = gold gradient, 🥈 Sage = silver, 🥉 Hunter = bronze).
- The leaderboard updates **in real time**: when another user's XP changes, their position animates (Framer Motion `layoutId` animation, smooth reorder).

### 37.4 Sharding Strategy for Scale

When the platform grows beyond ~1M users, a single Redis ZSET becomes a bottleneck. The sharding strategy:

1. **User ID hash → shard**: `shard = CRC32(user_id) % 128`.
2. **Write**: `ZADD lb:xp:alltime:{shard} {xp} {user_id}`.
3. **Read (Top 100)**: Fetch Top 100 from each of the 128 shards (12,800 entries total), merge‑sort in application memory (Python `heapq.merge`), return the final Top 100.
4. **Read (User's rank)**: `ZREVRANK lb:xp:alltime:{user_shard} {user_id}` + offset based on shard. For approximate global rank, use **Redis HyperLogLog** or a separate approximate counter.

### 37.5 Anti‑Gaming for Leaderboards

- **Minimum account age**: Accounts less than 24 hours old are excluded from public leaderboards.
- **Verified email required**: To appear on leaderboards, users must have a verified email address.
- **Suspicious activity flagging**: Users with XP/hour > 3σ are flagged and excluded from leaderboards pending review.
- **Bot detection**: Integration with FingerprintJS (browser fingerprinting) to detect multi‑account farming.
- **Appeal process**: Flagged users can appeal via the `gpu_reputation` dispute system, with a jury of 5 random validators (Kleros‑style) deciding.

---

## Feature 38: Bounty Board — Paid Analysis Requests

### 38.1 Architecture & Technology Stack

The Bounty Board follows the **AgentBounty architecture**: a decentralized bounty marketplace with "trustless escrow and on‑chain reputation tracking." The core insight is that "funds are securely held in smart contracts and only released once the Service Provider submits the 'Proof‑of‑Delivery'."

**Core technologies:**

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Frontend** | React (existing) + ethers.js v6 / web3.js for wallet interaction + `@solana/web3.js` if using Solana | Enables browser‑based wallet connection (MetaMask, Phantom, WalletConnect). |
| **Escrow Backend** | Your existing `gpu_escrow` Django app (centralized escrow) OR Solana smart contract (Anchor framework, Rust) for decentralized escrow | Centralized escrow is simpler and faster to implement; decentralized escrow provides trustlessness and is the long‑term vision. The architecture supports both: an abstraction layer (`EscrowProvider`) that can switch between centralized (PostgreSQL) and decentralized (Solana PDA) backends. |
| **Bounty Storage** | PostgreSQL `bounties` table + Elasticsearch for full‑text search | Bounties are stored in PostgreSQL. Elasticsearch indexes bounty titles, descriptions, tags, and skills for discoverability. |
| **Payment Processor** | Stripe Connect (for fiat) + Solana Pay / USDC on Solana (for crypto) | Dual payment rails: researchers can pay with credit card (Stripe) or crypto (USDC). Stripe Connect handles the marketplace payment flow (platform takes 5% fee). |
| **Notification** | WebSocket + Email (SendGrid) | Bounty creators are notified when a submission is received. Bounty hunters are notified when a bounty matching their skills is posted. |
| **Dispute Resolution** | Your existing `gpu_reputation` dispute system (Kleros‑style jury) | If a bounty creator rejects a submission, the hunter can escalate to a dispute. A jury of 5 randomly selected validators decides. |

### 38.2 Bounty Lifecycle

```
1. CREATE: Researcher posts bounty
   → POST /api/bounties { title, description, skills, deadline, reward_amount, reward_currency }
   → Funds locked in escrow (Stripe payment captured OR SOL transferred to escrow PDA)
   → Bounty status: "open"

2. CLAIM: Hunter claims the bounty
   → POST /api/bounties/{id}/claim
   → Bounty status: "in_progress"
   → Timer starts (deadline countdown)

3. SUBMIT: Hunter submits deliverable
   → POST /api/bounties/{id}/submit { deliverable_url, description, attachments }
   → Bounty status: "under_review"

4. REVIEW: Creator reviews submission
   → POST /api/bounties/{id}/review { decision: "accept" | "reject", feedback: "..." }
   
   4a. ACCEPT:
   → Escrow releases funds to hunter
   → Bounty status: "completed"
   → Hunter earns XP + reputation

   4b. REJECT:
   → Bounty status: "disputed"
   → Dispute resolution process starts
   → Jury of 5 validators reviews evidence
   → Majority vote decides outcome
   → Funds released accordingly

5. EXPIRE: Deadline passes without submission
   → Funds returned to creator
   → Bounty status: "expired"
```

### 38.3 Bounty Discovery & Matching

**Elasticsearch indexing for bounty search:**

```json
// PUT /bounties/_doc/{bountyId}
{
  "title": "Analyze attention patterns of BERT-base fine-tuned on MRPC",
  "description": "I need a detailed analysis of how attention heads specialize after fine-tuning...",
  "skills": ["attention_visualization", "bert", "nlp", "mechanistic_interpretability"],
  "reward_amount": 50.00,
  "reward_currency": "USD",
  "deadline": "2026-06-15T00:00:00Z",
  "creator_id": "uuid",
  "status": "open",
  "created_at": "2026-05-26T00:00:00Z"
}
```

**ML‑based matching (optional, Phase 2):** A sentence‑transformer model (`all‑MiniLM‑L6‑v2`) computes embeddings for bounty descriptions and user profiles. Bounties are recommended to users whose profile embedding has cosine similarity > 0.6 with the bounty embedding.

### 38.4 Reward Distribution

| Scenario | Creator Pays | Hunter Receives | Platform Fee | Jury Fee |
|----------|-------------|-----------------|--------------|----------|
| **Accept** | $50.00 | $47.50 (95%) | $2.50 (5%) | $0 |
| **Dispute → Hunter wins** | $50.00 | $42.50 (85%) | $2.50 (5%) | $5.00 (10%, split among 5 jurors) |
| **Dispute → Creator wins** | $50.00 | $0 (returned to creator minus fees) | $2.50 (5%) | $5.00 (10%) |
| **Expired** | $50.00 (returned) | $0 | $0 | $0 |

### 38.5 Bounty Board UI (Zone 8 Panel)

The bounty board in Zone 8 displays a compact list of "Open Bounties Matching Your Skills" (top 3, with a "View All" link).

```
┌──────────────────────────────────────────┐
│ 💰 Bounty Board                           │
├──────────────────────────────────────────┤
│                                          │
│  📊 Attention Analysis Needed            │
│     BERT‑base fine‑tuned on MRPC         │
│     💵 $50 · ⏰ 20 days left              │
│     [Claim]                              │
│     ────────────────────────────────     │
│  🔬 Cross‑Architecture Comparison        │
│     Galerkin vs Standard MHA             │
│     💵 $100 · ⏰ 15 days left             │
│     [Claim]                              │
│     ────────────────────────────────     │
│  🧩 Identify Redundant Heads             │
│     Compressed DistilBERT model          │
│     💵 $75 · ⏰ 8 days left               │
│     [Claim]                              │
│                                          │
│  [Post a Bounty]   [View All (42 open)]  │
└──────────────────────────────────────────┘
```

**Post a Bounty** opens a modal with a form (title, description, skills, reward amount, deadline). On submission, the creator is redirected to Stripe Checkout (or prompted to sign a Solana transaction) to fund the escrow.

---

## Integration Summary: Zone 8 ↔ All Other Zones

| Zone | How Zone 8 Integrates |
|------|----------------------|
| **Zone 1 (Context Bar)** | Level and XP displayed in user avatar dropdown; breadcrumb shows "Attention Lab › Level 8" |
| **Zone 2 (Visualization)** | Every pattern view awards XP; badges unlock premium visualization modes (3D view requires Level 3) |
| **Zone 3 (Math Panel)** | Exploring equation steps awards XP; "Math Explorer" badge for viewing all 4 computation steps |
| **Zone 4 (Control Panel)** | "Train with This" awards bonus XP; certain attention types require minimum level to access |
| **Zone 5 (Analytics)** | Running ablation/logit lens awards XP; "Ablation Expert" badge for 50 ablations |
| **Zone 6 (Interpretability)** | Publishing circuits awards XP; "Circuit Master" badge for 50 circuits |
| **Zone 7 (Social)** | Stars, comments, shares, forks all award XP; leaderboard shows top publishers/curators; bounties integrate with social profile |

The entire Zone 8 system is built on an event‑driven architecture where every meaningful user action emits a Kafka event. The XP engine, badge engine, leaderboard updater, and streak tracker all consume these events independently, enabling each subsystem to scale and evolve separately. The use of Redis Sorted Sets for leaderboards, a YAML‑based rule engine for badges, and a hybrid centralized/decentralized escrow for bounties ensures the system is both performant and flexible—ready for everything from a small research lab to a global platform with millions of users.