---
name: report-chapter-1-writer
description: "Write Chapter 1 (Introduction) using AUDIT_03, AUDIT_04, and docs/thesis_agents guides"
version: "1.0.0"
tags: ["neuralops", "report", "chapter-1", "introduction", "writing", "thesis_agents"]
---

# Skill: Chapter 1 (Introduction) Authoring and Empirical Framing

## When to use
Use this skill when composing, revising, or validating Chapter 1 (Introduction and General Overview) of the doctoral dissertation (`academic/thesis/810101504_main.tex` or `thesis_final/810101504_main.tex`). It enforces strict compliance with the agent guidelines in `docs/thesis_agents/` and the Persian writing rules in `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_01_CHAPTER_1_FOUNDATIONS.md`.
- Direct compliance with `docs/thesis_agents/AGENT_02_CHAPTER_1_RQS_CONTRIBUTIONS.md`.
- Compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target files: `academic/thesis/810101504_main.tex` (lines 238–500) or `thesis_final/chapters/chapter1.tex`.
- Remote GPU access: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9) with `/mnt/d/venv/bin/python3`.
- Read access to `results/registry/master_ledger.jsonl`.

## Canonical Audit Files Reference
Chapter 1 relies on empirical data across all six canonical audit files:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in Section 1.3 for RoBERTa-base recovery metrics ($52.06\% \rightarrow 73.62\%$).
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in Section 1.3 for singular value tail energy preservation ($+21.85\%$).
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in Section 1.1 and 1.2 for the Williams hardware roofline model (183.1 FLOP/Byte) and 203-module telemetry.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in Section 1.1 for quadratic memory exhaustion ($T=8192$ OOM cliff at 2.4 GB) and linear crossover thresholds.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in Section 1.5 for the six core system achievements and reproducible testbed environment disclosures.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced for mapping Chapter 1 drafting tasks to Agent 01 and Agent 02 roles and tracking line-number boundaries.

## Canonical Repository Invariants
Maintain canonical repository counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Section-by-Section Drafting Architecture

### Section 1.1 — The Memory Wall and Attention Bottleneck
Guided by `docs/thesis_agents/AGENT_01_CHAPTER_1_FOUNDATIONS.md`:
- Open with dynamic memory footprint challenges rather than abstract AI generalities.
- Present the total activation memory formula:
  $$M_{\text{act}} = B \cdot L \cdot N \cdot D \cdot \left( 2 + \frac{H \cdot N}{D} \right) \times \text{BytesPerElement}$$
- Show that for $N \in \{512, 1024, 2048, 4096, 8192\}$, memory escalates until uncompressed attention allocates $>2.4\text{ GB}$ per layer at $T=8192$ on RTX 4070 (12 GB GDDR6X), causing instant CUDA OOM faults (`AUDIT_04` Table 1).
- Integrate the Williams Roofline model from `AUDIT_03`, proving that standard attention arithmetic intensity saturates at $183.1\text{ FLOP/Byte}$ and operates firmly in the memory-bandwidth-bound regime ($255.8\text{ GB/s}$).
- Include Figure 1.1 (`fig_fla_scaling.pdf` or `seqlen_scaling_loglog.pdf`) demonstrating linear-softmax crossover at $N \approx 2048$.

```latex
% Canonical opening snippet for Section 1.1 conforming to AGENT_01
محدودیت اصلی در استنتاج و آموزش مدل‌های زبانی بزرگ، ناشی از مقیاس‌بندی غیرخطی مصرف حافظه است \cite{audit04}. 
در الگوریتم‌های توجه استاندارد \lr{Softmax}، ردپای فعال‌سازی‌ها \lr{(Activation Footprint)} با طول دنباله 
رابطهٔ درجه دو $\mathcal{O}(N^2)$ دارد. تله‌متری تجربی ثبت‌شده در جدول ۱ از مأخذ \cite{audit04} نشان می‌دهد 
که در طول توالی $T=8192$، لایه‌های توجه بدون فشرده‌سازی بیش از ۲٫۴ گیگابایت حافظهٔ پویا اشغال می‌کنند 
که در کارت‌های گرافیکی با حافظهٔ ۱۲ گیگابایت نظیر \lr{RTX 4070}، بلافاصله منجر به خطای سرریز حافظه 
\lr{(CUDA OOM)} می‌گردد. همچنین تحلیل شدت محاسباتی بر اساس مدل سقف کارایی ویلیامز \cite{audit03} 
اثبات می‌کند که این عملگر در نقطهٔ $183.1\text{ FLOP/Byte}$ اشباع شده و در ناحیهٔ محدود به پهنای باند حافظه 
\lr{(Memory-Bound)} گرفتار می‌ماند.
```

### Section 1.2 — Problem Statement: Architectural Fragmentation
Guided by `docs/thesis_agents/AGENT_01_CHAPTER_1_FOUNDATIONS.md`:
- Address the "203 modules without a unified framework" paradox: 131 attentions and 72 compressors exist across fragmented codebases with divergent APIs and unverified kernel backends (`AUDIT_03`).
- Define distribution drift (رانش توزیع) and activation subspace collapse following uncalibrated post-training surgery.

### Section 1.3 — Research Motivation: Principled Low-Rank Factorization
Guided by `docs/thesis_agents/AGENT_01_CHAPTER_1_FOUNDATIONS.md`:
- Contrast naive Eckart-Young-Mirsky truncated SVD ($52.06\%$ accuracy on RoBERTa-base SST-2 in `AUDIT_01`) with HNC-SVD ($73.62\%$ accuracy) and tail energy retention ($+21.85\%$ in `AUDIT_02`).

### Section 1.4 — Research Questions (RQ1 through RQ8)
Guided by `docs/thesis_agents/AGENT_02_CHAPTER_1_RQS_CONTRIBUTIONS.md`:
- Explicitly state RQ1 through RQ8 mapped to specific audit evidence tables:
  - RQ1 (Multi-Tier Acceleration) $\rightarrow$ `AUDIT_03` Table 1.
  - RQ2 (Linear Attention Crossover) $\rightarrow$ `AUDIT_04` Table 1.
  - RQ3 (Rank Compression Bounds) $\rightarrow$ `AUDIT_02` Table 2.
  - RQ4 (Distillation Synergy) $\rightarrow$ `AUDIT_01` Table 3.
  - RQ5 (Memory Cache Footprint) $\rightarrow$ `AUDIT_04` Table 2.
  - RQ6 (In-Place Surgery) $\rightarrow$ `AUDIT_01` Table 4.
  - RQ7 (Roofline Saturation) $\rightarrow$ `AUDIT_03` Section 6.
  - RQ8 (Forensic Traceability) $\rightarrow$ `AUDIT_05` Table 1.

```latex
\begin{table}[H]
\centering
\small
\caption{نگاشت سؤالات اصلی پژوهش به مأخذ ممیزی داده‌ها \cite{agents_guide}}
\label{tab:ch1_rq_audit_map}
\begin{tabularx}{\textwidth}{@{} l X l @{}}
\toprule
\textbf{سؤال پژوهش} & \textbf{هدف ارزیابی تجربی} & \textbf{مأخذ ممیزی} \\
\midrule
\lr{RQ1}: شتاب گسیل سه‌سطحی & دستیابی به شتاب $\ge 1.0\times$ در $203$ ماژول & \texttt{AUDIT\_03} \\
\lr{RQ2}: آستانه عطف خطی & تعیین نقطه تقاطع در $N \approx 2048$ توکن & \texttt{AUDIT\_04} \\
\lr{RQ3}: بقای انرژی طیفی & حفظ انرژی بالای $95\%$ در نسبت رتبه $0.5$ & \texttt{AUDIT\_02} \\
\lr{RQ4}: هم‌افزایی تقطیر ۶ ستون & برتری نسبت به واگرایی تکی \lr{KL} & \texttt{AUDIT\_01} \\
\lr{RQ5}: مقیاس بافر حالت & اثبات حافظه ثابت $\mathcal{O}(1)$ در مدل‌های بازگشتی & \texttt{AUDIT\_04} \\
\lr{RQ6}: جراحی بدون آموزش مجدد & پیوند درجا با حفظ شباهت کسینوسی بالای $0.98$ & \texttt{AUDIT\_01} \\
\lr{RQ7}: اشباع سقف کارایی & دستیابی به سقف پهنای باند $255.8\text{ GB/s}$ & \texttt{AUDIT\_03} \\
\lr{RQ8}: بازتولیدپذیری جرم‌شناختی & ثبت $2432$ اجرای تجربی با برچسب هش گیت & \texttt{AUDIT\_05} \\
\bottomrule
\end{tabularx}
\end{table}
```

### Mandatory Visual Analytics & Mechanistic Reasoning Mandate
Every figure and table MUST be accompanied by at least two to three substantive paragraphs of deep physical, algebraic, and architectural reasoning:
- **Root-Cause Hardware Analysis**: Detail microarchitectural stalls, SRAM capacity limits (e.g. 128 KB per SM), DRAM bandwidth bounds (e.g. 504 GB/s on RTX 4070), register spills, warp divergence, and L2 cache thrashing.
- **Mathematical Invariant Explanations**: Detail condition number explosion ($\kappa(\mathbf{S}_X) \gg 10^5$), Stiefel manifold curvature, rank collapse dynamics, Fisher information pullback, and discrete recovery derivative $\Delta(r)$.
- **Empirical Failure & Success Autopsies**: Explain exactly WHY an algorithm succeeded or failed at the hardware level, never merely reporting numbers without causal explanation.

### Strict Table Formatting Standards (English Math-Mode Digits & Zero Overflow)
- **Mandatory English Digits**: All digits, percentages, speedups, and dimensions in every table MUST be in English math mode `$ ... $` (e.g. `$94.50\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`, `$34.80\%$`, `$r=32$`). Persian digits in table cells are strictly prohibited to prevent XePersian decimal point flipping and reversed negative signs.
- **Widen Layout & Prevent Overflows**: Use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` with `\small` or `\footnotesize`, adjust `\tabcolsep` (e.g. `4pt` or `4.5pt`), and use `\arraystretch{1.2}` to ensure tables never exceed the text margins (`Zero Overfull \hbox`).

### Section 1.5 — Summary of Key Contributions
Guided by `docs/thesis_agents/AGENT_02_CHAPTER_1_RQS_CONTRIBUTIONS.md`:
- Detail the 6 canonical architectural achievements from `AUDIT_05` Section 1.

---

## Persian Writing & XePersian Discipline (`PERSIAN_WRITING_RULES.md`)

When drafting Chapter 1 in Persian XePersian, adhere to this strict compliance checklist:
- **`\lr{...}` Macro Constraints**:
  - Use for: `\lr{PyTorch}`, `\lr{CUDA}`, `\lr{Triton}`, `\lr{NeuralOps}`, `\lr{HNC-SVD}`, `\lr{SST-2}`.
  - FORBIDDEN inside math mode (`\begin{equation}`, `$ ... $`).
  - FORBIDDEN enclosing `\item`. Never write `\lr{\item ...}`.
- **نیم‌فاصله (ZWNJ `U+200C`)**:
  - Mandatory in: `مدل‌های`, `کِرنل‌های`, `لایه‌ای`, `بهینه‌سازی`, `پیش‌رو`, `بازگشتی`.
- **Persian Digits & Percentage**:
  - In Persian narrative text: **۱۳۱ مکانیزم توجه**, **۷۲ موتور فشرده‌سازی**.
  - In Persian sentences: use `٪`, never `%`.
- **Formal Academic Tone**:
  - Use standard Persian terminology approved by Tehran University (جراحی مدل, تقطیر دانش, دیوار حافظه).

---

## Verification
1. **Thesis Agent Adherence**:
   ```bash
   grep -q "sec:intro_story" academic/thesis/810101504_main.tex && echo "PASS: Section labels match AGENT_01"
   ```
2. **Audit Citations**:
   ```bash
   grep -c "AUDIT_0" academic/thesis/810101504_main.tex
   # Must be >= 15 across Chapter 1
   ```
3. **XeLaTeX Build**:
   ```bash
   cd academic/thesis && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must produce 0 fatal errors
   ```

---

## Common Pitfalls
- Ignoring `AGENT_01_CHAPTER_1_FOUNDATIONS.md` and using generic text without activation memory formulas.
- Ignoring research question mappings defined in `AGENT_02_CHAPTER_1_RQS_CONTRIBUTIONS.md`.
- Misusing `\lr{...}` inside LaTeX math environments, causing fatal compilation errors.
- Joining Persian compound words without ZWNJ (`مدلهای` instead of `مدل‌های`).
- Rounding exact telemetry metrics (e.g., altering `73.62%` to `74%`).
- Omitting provenance citation keys defined in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
