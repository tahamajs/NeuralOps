# 🔥 Phase 4 – AI‑Agent‑Orchestrated Training Platform

**Timeline:** 6 weeks  
**Prerequisite:** Phase 3 (Kubernetes, Ray Train, RBAC, Alerting, Inference) fully operational  
**Goal:** An autonomous, self‑driving training platform where a colony of specialized AI agents continuously monitors, diagnoses, schedules, and optimizes every training run — with zero human intervention required for routine operations. The platform becomes a living system that gets smarter with every job it runs.

---

## 📐 Conceptual Architecture: The Agent Operating System

```
                           ┌──────────────────────────────────────┐
                           │        HUMAN (Research Lead)          │
                           │    Sets goals: "Beat SOTA on GLUE"    │
                           └──────────────┬───────────────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT OPERATING SYSTEM (AOS)                 │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Cognitive │  │Scheduler │  │Diagnosti-│  │Safety &  │        │
│  │Agent     │  │Agent     │  │cian Agent│  │Audit     │        │
│  │(plans,   │  │(GPU allo-│  │(monitors,│  │Agent     │        │
│  │decides,  │  │cation,   │  │debugs,   │  │(approval │        │
│  │stopping) │  │priority) │  │recovers) │  │gates)    │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │              │              │              │              │
│       └──────────────┼──────────────┼──────────────┘              │
│                      ▼              ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Experience Library                         │ │
│  │  (persistent memory: what worked, what failed, why)         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                      │                                            │
│                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               Tool Server (MCP / REST)                       │ │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐  │ │
│  │  │Sub-│ │Read│ │Que-│ │Read│ │Modi-│ │Run │ │Git │ │Slack│  │ │
│  │  │mit │ │Logs│ │ry   │ │Code│ │fy   │ │Test│ │Ops │ │Noti-│  │ │
│  │  │Job │ │    │ │Pro- │ │    │ │Hype-│ │    │ │    │ │fy   │  │ │
│  │  │    │ │    │ │me-  │ │    │ │rpar-│ │    │ │    │ │     │  │ │
│  │  │    │ │    │ │theus│ │    │ │ams  │ │    │ │    │ │     │  │ │
│  │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

The AOS wraps the entire Phase 1‑3 platform and makes it autonomous. Agents never touch the cluster directly — they go through the MCP‑compliant Tool Server, which enforces safety policies, approval gates, and audit logging for every action.

---

## 1. 🏛️ The Agent Council — Five Specialized AI Agents

### 1.1 **Cognitive Agent** (the "PI")

| Capability | Implementation Detail |
|------------|----------------------|
| **Goal Decomposition** | Takes a high‑level research goal (e.g., "find the best compressed attention for MRPC under 50MB") and breaks it into experiment plans, sweep configurations, and stopping criteria  |
| **Optimal Stopping** | Uses Bayesian optimal stopping theory to decide "keep searching" vs. "we have the answer". Prevents infinite hyperparameter loops and wasted GPU hours |
| **Context Management** | Maintains a sliding context window with auto‑compaction at 170K tokens (ml‑intern pattern) . Retains only the most relevant observations |
| **Multi‑Step Reasoning** | Chain‑of‑thought deliberation before every action. Uses ReAct (Reasoning + Acting) loop |
| **Model**, data, task knowledge | Has access to the full HuggingFace model catalog, task descriptions, and historical results |

**Cadence:** Runs every 15‑30 minutes (or on‑demand when triggered by alerts). Awakens, reviews the Experience Library, plans next actions, delegates to Scheduler Agent.

---

### 1.2 **Scheduler Agent** (GPU Resource Manager)

| Capability | Implementation Detail |
|------------|----------------------|
| **Multi‑Agent RL Scheduling** | Uses MAIFS‑style multi‑agent RL where each worker node runs its own scheduling agent that coordinates via a dynamic coordination graph . Achieves 44% better makespan and 46% better JCT vs. heuristic schedulers  |
| **Fungible Resource Allocation** | Rather than fixed GPU counts, the agent learns each job's sensitivity to GPU count/type and can trade off "4×A100" vs. "2×H100" vs. "8×A100‑MIG‑sliced" |
| **Priority‑Aware Preemption** | Integrates with Kueue `WorkloadPriorityClass`. Decisions logged and auditable |
| **Gang Scheduling** | For Ray Train jobs that need N GPUs simultaneously, uses gang‑scheduling to prevent partial allocation that blocks everyone |
| **Straggler Mitigation** | Detects slow workers (NCCL heartbeat monitoring), decides: restart worker, migrate to different node, or reduce world size |

---

### 1.3 **Diagnostician Agent** (the "Doctor")

This is the most innovation‑dense agent. It uses the **Skill Framework** pattern from AMD's MaxText‑Slurm system  — structured runbooks that are not documentation but executable instruction files the agent follows step‑by‑step.

**Skills it can execute:**

| Skill | What it does |
|-------|-------------|
| `job‑log‑triage` | Reads tail of Slurm/K8s logs, classifies job status (running/blocked/failed/OOM), extracts step progress, recommends next action |
| `performance‑analysis` | Queries Prometheus TSDB for throughput decline patterns. Correlates GPU utilization, NCCL bandwidth, host CPU/memory, and network drops |
| `tsdb‑diagnosis` | Deep time‑series analysis: detects anomalous drops in tokens‑per‑GPU‑per‑second (TGS), isolates to model behavior vs. hardware fault vs. network partition |
| `OOM‑root‑cause` | Diagnoses out‑of‑memory: was it a memory leak? too large a batch? model config error? fragmentation? Suggests remediation |
| `NCCL‑timeout` | Checks NCCL environment, RDMA/IB status, switch logs. Proposes ring‑order change or NCCL parameter tuning |
| `checkpoint‑corruption` | Validates checkpoint integrity via structural inspection (safetensors header check) and lightweight inference test |
| `code‑introspection` | Reads the training Python source (via file‑system access), cross‑references with MaxText or Transformers framework code, traces errors to specific lines  |

**Cadence:** Continuously polls active jobs. On any anomaly (loss NaN, gradient overflow, NCCL timeout, OOM), is immediately invoked. For healthy jobs, runs every 5 minutes as a wellness check.

---

### 1.4 **Safety & Audit Agent** (the "Guardian")

| Capability | Implementation Detail |
|------------|----------------------|
| **Approval Gates** | Before any destructive action (cancel job, modify hyperparams, delete data), the agent pauses and requests explicit human approval via Slack/UI  |
| **Transactional Actions** | Every operation follows the apply→commit→revert pattern from the OS‑tuning agent model 【12†L14‑L17】. If an action fails mid‑execution, the state is rolled back |
| **Doom‑Loop Detection** | Watches for repeated tool call patterns (3+ identical calls in a row), injects a corrective prompt forcing the agent to try a different approach |
| **Audit Trail** | Every agent action (who, what, when, why) is logged immutably to a tamper‑proof append‑only log. Full reproducibility for compliance |
| **Policy Enforcement** | Reads organizational policies from a YAML config: max GPU‑hours per experiment, forbidden model types, mandatory checkpoint intervals |

---

### 1.5 **Experience Librarian** (the Memory)

Inspired by SIRIUS  and VIGIL's EmoBank 【15†L20‑L23】, this agent maintains the platform's long‑term memory.

| Capability | Implementation Detail |
|------------|----------------------|
| **Knowledge Graph** | Stores every experiment result as a structured triple: `(config, metrics, diagnosis)`. Enables queries like "what attention types work best for small batch sizes on MRPC?" |
| **Failure Database** | Every error + root cause + fix is stored. When a similar failure pattern is detected, the Diagnostician Agent retrieves the known fix instantly |
| **Curriculum Library** | Stores successful training recipes as reusable templates. New experiments can start from the nearest neighbor in the library rather than from scratch |
| **Decay & Aging** | Older experiences are weighted less (exponential decay with a long half‑life). Prevents the system from getting stuck on obsolete knowledge |
| **Cross‑Task Transfer** | Identifies patterns that transfer across tasks: "low‑rank attention works well for all pair‑classification tasks" → promotes that insight globally |

---

## 2. 🔌 Tool Server (MCP‑Compliant)

All agents access the platform through a **Model Context Protocol (MCP)** server — the same pattern used by Anthropic's OS‑tuning agent 【12†L12‑L16】 and the production‑grade agentic workflow guide 【18†L38‑L45】.

```
         Agent ( Claude / GPT / Gemini / Open‑source )
                         │
           ┌─────────────┼─────────────┐
           │   MCP Client (model‑specific) │
           └─────────────┼─────────────┘
                         │
              JSON‑RPC over WebSocket
                         │
           ┌─────────────▼─────────────┐
           │    MCP TOOL SERVER         │
           │  (Python, async, AuthZ)    │
           └─────────────┬─────────────┘
                         │
     ┌──────┬──────┬─────┼─────┬──────┬──────┐
     ▼      ▼      ▼     │     ▼      ▼      ▼
  Submit  Read   Query   │  Modify  Run   Slack
   Job    Logs  Prometh- │  Hyper-  Test  Notify
                 eus     │  params
```

**Tools exposed:**

| Tool Name | Access Level | Description |
|-----------|-------------|-------------|
| `submit_training_job` | Elevated (approval) | Creates and submits a training job with full config |
| `cancel_job` | Elevated (approval) | Gracefully stops a running job |
| `read_job_logs` | Read‑only | Tail logs from any job (last N lines or time range) |
| `query_prometheus` | Read‑only | Execute any PromQL query against the TSDB |
| `read_training_code` | Read‑only | View the actual Python training script being executed |
| `modify_hyperparameters` | Elevated (approval) | Dynamically change learning rate, batch size, etc. on a running job |
| `run_test_suite` | Read‑only | Execute the model's unit/integration tests on a checkpoint |
| `list_checkpoints` | Read‑only | List available checkpoints with metadata |
| `restore_from_checkpoint` | Elevated (approval) | Resume a failed job from a specific checkpoint |
| `update_experience_library` | Write | Record an observation, success, or failure |
| `query_experience_library` | Read‑only | Search past experiments for patterns |
| `send_slack_notification` | Write | Alert a human channel (throttled to avoid spam) |

**Safety design:**

- Every tool call is **authenticated** (agent identity verified).
- Every tool call is **authorized** against a policy engine (OPA/Rego rules).
- Every tool call is **logged** immutably (append‑only JSON‑lines audit log).
- Elevated tools require **human approval** via Slack interactive buttons or a web UI confirmation dialog.

---

## 3. 🔄 Autonomous Training Loop (How It All Fits Together)

This is the core workflow — the closed loop that runs indefinitely:

```
                     ┌──────────────────┐
                     │ Human sets goal   │
                     │ "Beat SOTA on     │
                     │  MNLI with <100MB │
                     │  model"           │
                     └────────┬─────────┘
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │                                                           │
   │  ╔══════════════════════════════════════════════════════╗  │
   │  ║          AUTONOMOUS EXPERIMENT LOOP                 ║  │
   │  ╚══════════════════════════════════════════════════════╝  │
   │                                                           │
   │  ┌─────────┐    ┌──────────┐    ┌──────────┐            │
   │  │Cognitive│───▶│ Scheduler│───▶│  Launch  │            │
   │  │ Agent   │    │ Agent    │    │ Training │            │
   │  │ (Plan)  │    │ (Alloc)  │    │  Job(s)  │            │
   │  └────▲────┘    └──────────┘    └────┬─────┘            │
   │       │                              │                   │
   │       │     ┌──────────┐    ┌───────▼──────┐            │
   │       │     │Experience│    │ Diagnostician│            │
   │       └─────│Library   │◀───│ Agent        │            │
   │       │     │          │    │ (Monitor)    │            │
   │       │     └────▲─────┘    └──────┬───────┘            │
   │       │          │                 │                     │
   │       │          │          ┌──────▼───────┐            │
   │       │          │          │   Training   │            │
   │       │          └──────────│   Running    │            │
   │       │                     │   (TF/PyT)   │            │
   │       │                     └──────┬───────┘            │
   │       │                            │                     │
   │       │                     ┌──────▼───────┐            │
   │       │                     │  Safety &    │            │
   │       └─────────────────────│  Audit Agent │            │
   │                             └──────────────┘            │
   │                                                           │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Human approves   │
                     │ final model and  │
                     │ receives summary │
                     │ report           │
                     └──────────────────┘
```

### Detailed Step‑by‑Step

1. **Human sets a research goal** in natural language: "Find the best compressed attention for the MNLI task, model must be under 100 MB, accuracy within 1% of full model."

2. **Cognitive Agent** activates:
   - Parses the goal into a structured experiment plan.
   - Queries the Experience Library: "Has anyone compressed MNLI before? What attention types worked?"
   - Generates a sweep config: 5 attention types × 4 ranks × Bayesian search on lr/batch size.
   - Sets a stopping criterion: stop when 3 consecutive trials fail to improve the best result by >0.1%.

3. **Scheduler Agent** activates:
   - Assesses cluster state: 4 A100‑40GB GPUs available, 2 jobs already running, queue depth of 3.
   - Uses its trained MAIFS policy to allocate: 2 GPUs for this sweep (non‑preemptible), 2 GPUs for background jobs.
   - Sets priority: medium (research sweep).

4. **Training launches** — sweep begins. Each trial is a Celery task that runs the framework's `run_experiment.py` pipeline.

5. **Diagnostician Agent** monitors continuously:
   - Every 5 minutes: polls Prometheus for all active jobs.
   - Trial #3 shows loss diverging at step 200: NaN detected.
   - Agent reads the training logs, identifies the issue: learning rate too high with this attention type.
   - **Self‑healing action:** modifies the hyperparameter config for trial #3 (halves LR), marks the trial for restart.
   - Records in Experience Library: `(aurora_attention, lr=5e-5, mnli) → NaN at step 200. Root cause: lr too high. Fix: lr ≤ 2e-5`.

6. **Safety & Audit Agent** ensures all modifications are logged and (if configured) requests Slack approval for the hyperparameter change.

7. **Loop continues** until the Cognitive Agent's stopping criterion is met. The best model is automatically registered in the Model Registry, tagged `production‑ready`.

8. **Human receives** a Slack summary:
   ```
   ✅ MNLI compression experiment complete.
   Best: aurora_attention, rank=48, accuracy=0.863 (baseline=0.870, difference=-0.007)
   Model size: 82 MB. 12 trials run, 3 auto‑recovered, 0 human interventions needed.
   Model registered as: mnli‑aurora‑v3
   ```

---

## 4. 🧠 Self‑Improving Curriculum (Meta‑Learning)

Beyond reactive monitoring, the system actively **learns to train better** over time.

### 4.1 Auto‑Curriculum Generation

**Inspiration:** LAIMARK  and ml‑intern  show that LLMs can generate their own training data and improve themselves.

How it works in our system:

- When the platform has been idle for >1 hour (weekend, overnight), the Cognitive Agent activates a **self‑study mode**.
- It generates synthetic training curricula — small, fast experiments designed to probe the limits of the system: "What happens if I set learning_rate to 1e-7?" "Can I train MNLI with only 100 examples?"
- Each experiment runs for 5 minutes (the Karpathy autoresearch pattern) .
- Results feed back into the Experience Library.

### 4.2 Cross‑Task Knowledge Transfer

The Experience Librarian identifies patterns that generalize:

- "All attention types with rank < 16 fail on pair‑classification tasks" → global rule.
- "Low‑rank attention works well for all tasks except CoLA" → task‑specific exception.

These patterns become **policy rules** that the Cognitive Agent follows automatically, without needing to rediscover them.

### 4.3 Metacognitive Self‑Assessment

**Inspiration:** MAGELLAN metacognitive framework .

After each action, the Cognitive Agent self‑evaluates:

```
Was my decision correct?
Did the experiment improve the metric?
Was my stopping decision optimal?
```

This self‑assessment feeds into a **confidence calibration** system. Over time, the agent learns to trust its own judgment on certain types of decisions (e.g., "I'm 95% accurate at detecting when a learning rate is too high") while remaining cautious on others (e.g., "I'm only 60% accurate at choosing attention types for new domains").

---

## 5. 🔬 Advanced Agent Capabilities

### 5.1 Multi‑Agent Collaborative Optimization

**Inspiration:** MAT‑Agent (NeurIPS 2025)  and OptiMindTune .

For complex training optimization, the system spawns a **collaborative swarm** of 3 sub‑agents:

| Sub‑Agent | Role |
|-----------|------|
| **Recommender Agent** | Proposes hyperparameter configurations, attention types, and model architectures based on the Experience Library |
| **Evaluator Agent** | Assesses each proposal against the goal: estimates expected accuracy, compute cost, and risk of failure |
| **Decision Agent** | Uses a multi‑armed bandit (UCB + ε‑greedy) to select which proposal to execute next, balancing exploration vs. exploitation  |

The three agents share observations via a shared blackboard. Decisions are made hierarchically: Evaluator filters proposals, Decision chooses the execution order.

### 5.2 On‑the‑Fly Architecture Search

While a model is training, the Diagnostician Agent can decide to **modify the architecture mid‑training**:

- If a layer shows near‑zero attention entropy → merge it with the adjacent layer (layer fusion).
- If a layer's singular value spectrum is highly concentrated → reduce its rank (dynamic rank adjustment).
- If validation accuracy plateaus for 3 epochs → trigger a "growth phase" — temporarily increase rank, add dropout, or switch attention type.

**Inspiration:** NAS‑DQN (dynamic architecture reconfiguration during training)  and Agentic Supernet (dynamic agent team composition based on task difficulty) 【6†L44‑L48】.

### 5.3 Full‑Autonomy Overnight Mode

**Inspiration:** Karpathy's autoresearch pattern  and HuggingFace ml‑intern .

When enabled by the user, the system enters **full‑autonomy mode**:

- The human goes home at 6 PM with a goal: "find me the best compressed model for all 8 GLUE tasks by tomorrow."
- The Cognitive Agent plans and executes a massive sweep across tasks × attention types × ranks × hyperparameters.
- The Scheduler Agent manages the GPU queue, preempts lower‑priority jobs, and coordinates across 8‑64 GPUs.
- The Diagnostician Agent fixes any failures autonomously (within safety guardrails).
- The Experience Librarian records every result.
- At 8 AM, the human returns to a comprehensive report: best models per task, a summary of what was tried, and automatically registered models ready for deployment.

---

## 6. 🏗️ Implementation Schedule (6 Weeks)

### Week 1–2: Foundation — MCP Tool Server + Experience Library

1. **Design the MCP Tool Server** using FastAPI + JSON‑RPC over WebSocket.
2. **Implement all 12 tools** listed in the Tool Server section.
3. **Build the Experience Library** using PostgreSQL with JSONB for flexible queries.
4. **Implement the Safety & Audit Agent** — approval gates, transactional actions, audit logging.
5. **Deploy and test** with a single agent (Claude API) manually invoking tools via the MCP server.

### Week 3–4: Agent Council — Cognitive + Scheduler + Diagnostician

6. **Implement the Cognitive Agent** — goal decomposition, experiment planning, stopping criteria.
7. **Implement the Diagnostician Agent** — skill framework (5 initial skills), Prometheus integration, log analysis.
8. **Implement the Scheduler Agent** — MAIFS integration with Kueue (or simple priority‑based MVP).
9. **Integrate all agents** — they communicate through the Experience Library (shared memory) and Tool Server (shared actions).
10. **Build the Slack integration** for notifications and approval requests.

### Week 5: Autonomy Loop + Self‑Improvement

11. **Wire the autonomous experiment loop** — Cognitive → Scheduler → Launch → Diagnostician → Experience → Cognitive.
12. **Implement auto‑curriculum generation** — the system generates and runs its own experiments during idle periods.
13. **Implement metacognitive self‑assessment** — confidence calibration for agent decisions.
14. **End‑to‑end test:** give the system a goal, let it run for 24 hours, verify it makes progress without human intervention.

### Week 6: Multi‑Agent Optimization + Full Autonomy

15. **Implement the multi‑agent collaborative optimizer** (Recommender + Evaluator + Decision).
16. **Implement on‑the‑fly architecture modifications** — dynamic rank adjustment, layer fusion.
17. **Implement full‑autonomy overnight mode** with comprehensive morning summary.
18. **Build the Agent Dashboard UI** — React components showing agent activity, decision rationale, approval requests, and system health.

---

## 7. 🎨 Agent Dashboard UI (React)

New pages and components added to the existing React frontend:

| Page | Route | Description |
|------|-------|-------------|
| **Agent Console** | `/agents` | Real‑time view of all five agents — what each is doing right now, last thoughts, pending approvals |
| **Agent Timeline** | `/agents/timeline` | Scrollable event log showing every agent action with timestamps, tool calls, and outcomes |
| **Approval Queue** | `/agents/approvals` | Human approval requests — approve/reject with one click |
| **Experience Explorer** | `/agents/experience` | Search and browse the Experience Library — what has the system learned? |
| **Autonomy Settings** | `/agents/settings` | Configure: autonomy level (supervised / semi‑autonomous / full), approval thresholds, GPU budget |

**Key Components:**

| Component | Description |
|-----------|-------------|
| `AgentCard` | Shows agent name, status (active/idle/error), last action, next scheduled wake‑up |
| `ThoughtBubble` | Expandable chain‑of‑thought from the Cognitive Agent — shows its reasoning in real‑time |
| `ApprovalCard` | Shows the proposed action, the agent's justification, risk level (low/medium/high), and Approve/Reject buttons |
| `ExperienceGraph` | Force‑directed graph visualization of the knowledge graph — nodes are experiments/diagnoses, edges are relationships |
| `AutonomySlider` | Per‑agent control: manual → assisted → semi‑autonomous → full autonomous |
| `NightlyReport` | Auto‑generated morning summary — what happened overnight, best results, issues encountered |

---

## 8. 🔮 What Phase 4 Delivers

At the end of Phase 4, you have an **autonomous, self‑improving training platform** where:

- A **Cognitive Agent** plans experiments, sets stopping criteria, and decides the best path to the research goal.
- A **Diagnostician Agent** monitors every training run, detects failures in real‑time, identifies root causes using structured skills, and automatically recovers — saving hours of human debugging per week.
- A **Scheduler Agent** allocates GPU resources optimally using multi‑agent reinforcement learning, achieving 44% better throughput than static schedulers.
- A **Safety & Audit Agent** ensures every autonomous action is logged, justified, and (for dangerous operations) approved by a human.
- An **Experience Library** accumulates knowledge over time, so the platform gets smarter with every experiment — never repeating past mistakes.

The human's role shifts from **operator** (manually configuring, monitoring, debugging every run) to **research director** (setting goals, approving critical decisions, reviewing results).