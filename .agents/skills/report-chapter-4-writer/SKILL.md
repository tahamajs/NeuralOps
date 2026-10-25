---
name: report-chapter-4-writer
description: "Write Chapter 4 (Results) using ALL 5 AUDIT files and docs/thesis_agents guides"
version: "1.0.0"
tags: ["neuralops", "report", "chapter-4", "results", "benchmarks", "forensics", "thesis_agents"]
---

# Skill: Chapter 4 (Results) Empirical Synthesis and Forensic Evaluation

## When to use
Use this skill when drafting, verifying, or synthesizing Chapter 4 (Empirical Results and Performance Evaluation) of the doctoral dissertation (`academic/thesis/810101504_main.tex` or `academic/thesis/chapters/chapter04_*.tex`). It enforces strict compliance with the agent guidelines in `docs/thesis_agents/` and the Persian writing rules in `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_12_CHAPTER_4_MACRO_TELEMETRY.md`.
- Direct compliance with `docs/thesis_agents/AGENT_13_CHAPTER_4_ATTENTION_EVALUATION.md`.
- Direct compliance with `docs/thesis_agents/AGENT_14_CHAPTER_4_COMPRESSION_DISTILLATION.md`.
- Direct compliance with `docs/thesis_agents/AGENT_15_CHAPTER_4_KERNELS_SCALING.md`.
- Direct compliance with `docs/thesis_agents/AGENT_16_HARDWARE_FORENSICS_SAFETY.md`.
- Compliance with `AGENT_26`, `AGENT_27`, `AGENT_28`, and `AGENT_30_CHAPTER4_PRACTICAL_GUIDE.md`.
- Target files: `academic/thesis/810101504_main.tex` and `academic/thesis/chapters/`.
- Remote GPU access: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9) with `/mnt/d/venv/bin/python3`.
- Read access to `results/registry/master_ledger.jsonl`.

## Canonical Audit Files Reference
Chapter 4 serves as the central empirical nexus and draws heavily from all six canonical audit files:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in Section 4.3 for the 5-stage ablation study, KD Regimes 0 through 5 comparison tables, RoBERTa-base SST-2 rank degradation curves, and Ridge calibration sweeps.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in Section 4.3 for the 72-module compression engine evaluation, singular value spectral energy decay, and HNC vs Naive truncation comparisons.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in Section 4.1 and 4.2 for the comprehensive 203-module telemetry table, 131 attention operator latencies, and 108 CUDA C++ kernel execution timings.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in Section 4.4 for sequence length scaling benchmarks ($N \in [128, 8192]$), KV cache memory footprint allocations, and linear crossover dynamics.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in Section 4.5 for the 6 deep hardware forensics case studies, root-cause analyses, and forensic testbed execution logs.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced for coordinating writing duties across Agent 12–16, Agent 26–30 roles, and adhering to empirical section boundaries.

## Canonical Repository Invariants
Ensure canonical repository counts are maintained:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Section-by-Section Drafting Architecture

### Section 4.1 — Macro Telemetry Evaluation (203 Modules)
Guided by `docs/thesis_agents/AGENT_12_CHAPTER_4_MACRO_TELEMETRY.md` and `AGENT_28`:
- Present the macro performance envelope across all 203 unified operators (131 attention mechanisms + 72 compression engines) measured on NVIDIA RTX 4070 (SM 8.9).
- **Table 4.1**: Master 203-module performance registry (must include all 203 rows using `longtable`, reporting Module Name, Category, Family, Latency, Speedup, Diff, and VRAM).
- Integrate Figures:
  - Figure 4.1: Compression Pareto frontier (`figures/pareto_compressor.pdf`).
  - Figure 4.2: Speedup distribution histogram across all 203 modules (`figures/speedup_histogram.pdf`).

### Section 4.2 — Attention Zoo Performance Breakdown (131 Mechanisms)
Guided by `docs/thesis_agents/AGENT_13_CHAPTER_4_ATTENTION_EVALUATION.md`:
- Contrast the empirical behavior of all 7 attention families under standardized batch size $B=16$ and sequence length $S=2048$.
- **Table 4.2**: 131-row attention benchmark table detailing kernel backend execution, numerical deviation against PyTorch reference, and autograd gradient certification.
- Integrate Figure 4.3: 5-dimensional family radar chart (`figures/family_radar.pdf`).

### Section 4.3 — Compression Paradigms & Multi-Pillar Distillation
Guided by `docs/thesis_agents/AGENT_14_CHAPTER_4_COMPRESSION_DISTILLATION.md` and `AGENT_26`:
- Detail the multi-stage compression ablation study and distillation recovery progression:
- **Table 4.3**: 5-stage architectural ablation table (Baseline, SVD Truncation, Whitening, Ridge Calibration, Distillation Healing).
- **Table 4.4**: Systematic evaluation across KD Regimes 0 through 5 (All numbers MUST be English math-mode digits):

```latex
\begin{table}[H]
\centering
\small
\caption{ارزیابی مقایسه‌ای رژیم‌های تقطیر ۰ تا ۵ در مدل \lr{RoBERTa-base} بر روی مجموعه داده \lr{SST-2} \cite{audit01}}
\label{tab:kd_regimes_results}
\begin{tabular}{llrrrr}
\toprule
\textbf{رژیم تقطیر} & \textbf{ستون‌های فعال تابع زیان} & \textbf{دقت (٪)} & \textbf{\lr{PPL}} & \textbf{بقای کارایی} & \textbf{مأخذ} \\
\midrule
رژیم ۰ & برش ساده بدون وزن‌دهی & $52.06\%$ & $28.4$ & $57.8\%$ & \texttt{AUDIT\_01} \\
رژیم ۱ & واگرایی \lr{KL} لاجیت‌ها & $61.42\%$ & $22.1$ & $68.1\%$ & \texttt{AUDIT\_01} \\
رژیم ۲ & + خطای میانگین مربعات ویژگی‌ها & $66.85\%$ & $18.7$ & $74.2\%$ & \texttt{AUDIT\_01} \\
رژیم ۳ & + شباهت کسینوسی نقشه‌های توجه & $69.90\%$ & $16.5$ & $77.5\%$ & \texttt{AUDIT\_01} \\
رژیم ۴ & + زیان طیفی مقادیر تکین $\mathcal{L}_{\sigma}$ & $71.84\%$ & $15.2$ & $79.7\%$ & \texttt{AUDIT\_01} \\
رژیم ۵ & + تراز کوواریانس و تباینی & $73.62\%$ & $14.1$ & $81.7\%$ & \texttt{AUDIT\_01} \\
\bottomrule
\end{tabular}
\end{table}
```

- **Table 4.5**: RoBERTa-base SST-2 compression recovery across ranks $r \in [16, 32, 64, 128]$.
- Integrate Figures from `AGENT_29_FIGURE_GENERATOR.md`:
  - Figure 4.4: KD regime comparative bar chart (`fig_ablation_pipeline_5step.pdf`).
  - Figure 4.5: RoBERTa compression rank decay curve (`fig_hnc_4way_ablation.pdf`).
  - Figure 4.6: Distillation transfer fidelity heatmap (`fig_6pillar_heatmap.pdf`).

### Mandatory Visual Analytics & Mechanistic Reasoning Mandate
Every figure and table MUST be accompanied by at least two to three substantive paragraphs of deep physical, algebraic, and architectural reasoning:
- **Root-Cause Hardware Analysis**: Detail microarchitectural stalls, SRAM capacity limits (e.g. 128 KB per SM), DRAM bandwidth bounds (e.g. 504 GB/s on RTX 4070), register spills, warp divergence, and L2 cache thrashing (e.g. Chunk GLA 48 MB footprint overflowing the 36 MB hardware cache).
- **Mathematical Invariant Explanations**: Detail condition number explosion ($\kappa(\mathbf{S}_X) \gg 10^5$), Stiefel manifold curvature, rank collapse dynamics, Fisher information pullback, and discrete recovery derivative $\Delta(r)$.
- **Empirical Failure & Success Autopsies**: Explain exactly WHY an algorithm succeeded or failed at the hardware level, never merely reporting numbers without causal explanation.

### Strict Table Formatting Standards (English Math-Mode Digits & Zero Overflow)
- **Mandatory English Digits**: All digits, percentages, speedups, and dimensions in every table MUST be in English math mode `$ ... $` (e.g. `$94.50\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`, `$34.80\%$`, `$r=32$`). Persian digits in table cells are strictly prohibited to prevent XePersian decimal point flipping and reversed negative signs.
- **Widen Layout & Prevent Overflows**: Use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` with `\small` or `\footnotesize`, adjust `\tabcolsep` (e.g. `4pt` or `4.5pt`), and use `\arraystretch{1.2}` to ensure tables never exceed the text margins (`Zero Overfull \hbox`).
- **Beautification**: Use clean `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`) with centered numerical columns and left-aligned text descriptions.

### Section 4.4 — Kernel Sequence Length Scaling Dynamics
Guided by `docs/thesis_agents/AGENT_15_CHAPTER_4_KERNELS_SCALING.md` and `AGENT_27`:
- Analyze latency and peak memory allocation scaling across token sequence lengths $N \in [128, 8192]$.
- **Table 4.6**: Sequence scaling latency matrix comparing FlashAttention-2, Linear Attention, RetNet, and Mamba SSM.
- Integrate Figure 4.7: Log-log sequence scaling curve with $N \approx 2048$ crossover annotation (`fig_fla_scaling.pdf`).

### Section 4.5 — Hardware Forensics and Safety Constraints
Guided by `docs/thesis_agents/AGENT_16_HARDWARE_FORENSICS_SAFETY.md` and `AGENT_30_CHAPTER4_PRACTICAL_GUIDE.md`:
- Document the 6 anomalous kernel behavior case studies, deadlock triage, and 6 safety dispatch rules:
  1. Case 1: Selective SSM numerical NaN explosion under unbounded exponent scaling.
  2. Case 2: Out-of-bounds shared memory indexing in Tsallis 1.5-Entmax block tiling.
  3. Case 3: GDDR6 memory bus bandwidth throttling under uncoalesced global strides.
  4. Case 4: FP16 intermediate register overflow in multi-head latent projection GEMMs.
  5. Case 5: Warp-divergence stalls in causal splash attention prefix masks.
  6. Case 6: Numerical divergence in Cholesky decomposition on non-positive-definite covariance matrices.

```latex
\begin{table}[H]
\centering
\small
\caption{کالبدشکافی سخت‌افزاری ۶ ناهنجاری زمان اجرا و راهکارهای مهندسی پایدارسازی \cite{audit05}}
\label{tab:hardware_forensics_summary}
\begin{tabular}{clllc}
\toprule
\textbf{مورد} & \textbf{عملگر درگیر} & \textbf{نقص مشاهده‌شده} & \textbf{علت ریشه‌ای و اصلاح معماری} & \textbf{مأخذ} \\
\midrule
۱ & \texttt{selective\_ssm} & تولید مقادیر \lr{NaN} در گسیل & عدم محدودسازی توان $\rightarrow$ مهار عددی $e^{\Delta A}$ & \texttt{AUDIT\_05} \\
۲ & \texttt{tsallis\_15\_entmax} & خطای دسترسی غیرمجاز حافظه & سرریز اندیس کاشی $\rightarrow$ افزودن گارد مرزی & \texttt{AUDIT\_05} \\
۳ & \texttt{differential\_attn} & افت شدید کارایی ($>25$ میلی‌ثانیه) & گام‌های نامتقارن گلوبال $\rightarrow$ ترانهش در حافظه اشتراکی & \texttt{AUDIT\_05} \\
۴ & \texttt{mla\_attention} & واگرایی گرادیان در آزمون صحت & سرریز ثبات‌های \lr{FP16} $\rightarrow$ ارتقای انباشت به \lr{FP32} & \texttt{AUDIT\_05} \\
۵ & \texttt{adasplash\_attn} & قفل زمانی تارها \lr{(Warp Stall)} & انشعاب شرطی ماسک $\rightarrow$ برداری‌سازی کامل & \texttt{AUDIT\_05} \\
۶ & \texttt{hnc\_svd\_cholesky} & شکست تجزیه چولسکی & کوواریانس نامعین $\rightarrow$ میرایی قطری $\epsilon I$ & \texttt{AUDIT\_05} \\
\bottomrule
\end{tabular}
\end{table}
```

- Implement the 6 dispatch safety rules (`subsec:safety_rules`) and hardware constraint matrix (`subsec:practical_constraint_matrix`).

---

## Persian Writing & XePersian Discipline (`PERSIAN_WRITING_RULES.md`)
- Use `\lr{...}` for English tool/model names: `\lr{RoBERTa}`, `\lr{SST-2}`, `\lr{FlashAttention-2}`, `\lr{CUDA}`, `\lr{PyTorch}`.
- NEVER use `\lr{...}` inside math environments (`\begin{equation}`, `$ ... $`).
- NEVER wrap `\item` with `\lr{...}`.
- Mandatory نیم‌فاصله (ZWNJ `U+200C`) in all compound words (`مدل‌های`, `لایه‌ای`, `بهینه‌سازی`).
- Persian numbers in narrative text (**۱۳۱ مکانیزم توجه**, **۷۲ موتور فشرده‌سازی**) and Persian percent `٪`.

---

## Verification
1. **Thesis Agent Adherence**:
   ```bash
   grep -q "subsec:hardware_limits" academic/thesis/chapters/chapter04_*.tex && echo "PASS: Section labels match AGENT_30"
   ```
2. **Audit Citations**:
   ```bash
   grep -c "AUDIT_0" academic/thesis/chapters/chapter04_*.tex
   # Must be >= 20 across Chapter 4
   ```
3. **XeLaTeX Build**:
   ```bash
   cd academic/thesis && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must produce 0 fatal errors
   ```

---

## Common Pitfalls
- Quoting results from unverified preliminary scratch files instead of the canonical `AUDIT_01`–`AUDIT_05` documents.
- Omitting the 6 deep forensic investigations from `AGENT_16` and `AGENT_30`.
- Including fewer than 8 figures from `AGENT_29`, violating publication visualization requirements.
- Misusing `\lr{...}` inside math mode or wrapping `\item` in `\lr`.
