---
name: report-appendix-writer
description: "Write appendices (catalogs, proofs, additional tables) from audit data and docs/thesis_agents guides"
version: "1.0.0"
tags: ["neuralops", "report", "appendix", "catalog", "proofs", "longtable", "thesis_agents"]
---

# Skill: Appendix Authoring and Exhaustive Catalog Generation

## When to use
Use this skill when generating, formatting, or verifying the formal appendices of the doctoral dissertation (`academic/thesis/chapter_all_modules_appendix.tex`). It establishes the methodology for compiling the exhaustive 131-attention catalog (Appendix A), the 72-compressor catalog (Appendix B), the 108 CUDA C++ kernel execution catalog (Appendix C), and individual module monographs in `academic/thesis/attentions_latex/individual_modules/` according to Agent 19.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_19_APPENDICES_CATALOG_203.md`.
- Full compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target file: `academic/thesis/chapter_all_modules_appendix.tex`.
- Access to raw operator registries in `neuralops/registry/attention_registry.py` and `compressor_registry.py`.
- Strict exclusion of aliases, duplicate entries, and early uncalibrated runs.

## Canonical Audit Files Reference
The appendices serve as the exhaustive reference ledger for the manuscript, drawing full data from all six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in Appendix D for the complete analytical proof of Theorem 10 (Distillation Healing Error Contraction).
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in Appendix B for the full 72-row compressor catalog and in Appendix D for the proofs of Theorems 1, 2, and 3.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in Appendix A for the full 131-row attention catalog, in Appendix C for the 108 CUDA kernel execution table, and in Appendix D for Theorems 4 and 5.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in Appendix A and C for sequence length latency benchmarks across scaling horizons.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in Appendix A, B, and C for module verification certifications and testbed hardware specs.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced for appendix structure governance, author roles under Agent 19, and provenance routing.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Appendix Architecture and Content Specifications

### Appendix A — The Master 131-Attention Operator Catalog (Agent 19)
Target: `academic/thesis/chapter_all_modules_appendix.tex` (`app:all_131_attentions`).
- Compile every single registered attention operator into a multi-page LaTeX `longtable`.
- **Table A.1**: Must contain exactly 131 rows without omissions. Every row must present:
  1. Row index in English math mode (`$1$` to `$131$`).
  2. Canonical Module Name wrapped in `\lr{\texttt{...}}`.
  3. Architectural Family (Exact Softmax, Recurrent SSM, Linear, Low-Rank, Sparse, Differential, Hybrid).
  4. Time Complexity in standard math mode ($\mathcal{O}(N^2 d)$ or $\mathcal{O}(N d)$).
  5. Memory Complexity ($\mathcal{O}(N d)$ or $\mathcal{O}(d^2)$).
  6. Final Certification Status (`\lr{PASS\_FAST}` or `\lr{PASS\_SLOW}`).
  7. Optimal Execution Backend (`\lr{Native CUDA}`, `\lr{Triton JIT}`, or `\lr{PyTorch}`).

```latex
\chapter{پیوست الف: کاتالوگ و تله‌متری جامع ۱۳۱ مکانیزم توجه}
\label{app:all_131_attentions}

\begin{longtable}{@{}clllccr@{}}
\caption{کاتالوگ جامع ۱۳۱ مکانیزم توجه ثبت‌شده در پلتفرم به تفکیک خانواده و مشخصات محاسباتی}
\label{tab:app_all_131_attentions} \\
\toprule
\textbf{ردیف} & \textbf{شناسه رجیستری} & \textbf{خانوادهٔ معماری} & \textbf{پیچیدگی زمانی} & \textbf{پیچیدگی حافظه} & \textbf{وضعیت} & \textbf{بک‌اند برتر} \\
\midrule
\endfirsthead
\toprule
\textbf{ردیف} & \textbf{شناسه رجیستری} & \textbf{خانوادهٔ معماری} & \textbf{پیچیدگی زمانی} & \textbf{پیچیدگی حافظه} & \textbf{وضعیت} & \textbf{بک‌اند برتر} \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{\textit{ادامه در صفحهٔ بعد...}} \\
\bottomrule
\endfoot
\bottomrule
\endlastfoot

$1$  & \lr{\texttt{flash\_attention\_3}} & \lr{Fused}    & $\mathcal{O}(N^2 d)$   & $\mathcal{O}(N d)$ & \lr{PASS\_FAST} & \lr{Native CUDA} \\
$2$  & \lr{\texttt{cosformer}}          & \lr{Linear}   & $\mathcal{O}(N d^2)$   & $\mathcal{O}(N d)$ & \lr{PASS\_FAST} & \lr{Triton JIT} \\
$3$  & \lr{\texttt{gated\_delta\_net}}   & \lr{SSM}      & $\mathcal{O}(N d)$     & $\mathcal{O}(N d)$ & \lr{PASS\_FAST} & \lr{Triton JIT} \\
$4$  & \lr{\texttt{native\_sparse\_attn}}& \lr{Sparse}   & $\mathcal{O}(N \sqrt{N})$ & $\mathcal{O}(N k)$ & \lr{PASS\_FAST} & \lr{Native CUDA} \\
$5$  & \lr{\texttt{deepseek\_mla}}       & \lr{Low-Rank} & $\mathcal{O}(N r d)$   & $\mathcal{O}(N r)$ & \lr{PASS\_SLOW} & \lr{PyTorch} \\
% ... تمام ۱۳۱ ردیف باید درج گردند
\end{longtable}
```

### Strict Quality Standards: Digits, Overflows & Monographs
1. **Mandatory English Math-Mode Digits**: Every number, index, latency, speedup, accuracy, and VRAM quantity across all appendix tables MUST be written in English digits in math mode `$ ... $` (e.g. `$1$`, `$131$`, `$94.50\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`).
2. **Zero Overfull \hbox**: Wide tables must use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` with `\small` or `\footnotesize` and `\tabcolsep` tuning to eliminate text wrapping failures and margin overflows.
3. **Attention & Compressor Monographs**: Half-page structured analytical monographs for archetypal modules covering exact formulations, asymptotic complexities, KV-cache scaling footprints, SRAM/DRAM suitability, and empirical ablation profiles.
4. **Deep Mechanistic Reasoning**: Extensive physical and microarchitectural analysis explaining the root causes behind runtime behaviors.

### Appendix B — The Master 72-Model Compressor Catalog (Agent 19)
Target: `academic/thesis/chapter_all_modules_appendix.tex` (`app:all_72_compressors`).
- Compile all 72 registered post-training model compression engines.
- **Table B.1**: Must contain exactly 72 rows without truncation. Every row must detail:
  1. Row index in English math mode (`$1$` to `$72$`).
  2. Engine Identifier in `\lr{\texttt{...}}`.
  3. Compression Paradigm (Low-Rank, Pruning, Quantization, Distillation, Hybrid).
  4. Compression Factor in English math mode ($2.00\times$, $4.00\times$, $8.00\times$).
  5. Post-transplant relative weight reconstruction error in English math mode ($\|\mathbf{W} - \widehat{\mathbf{W}}\|_F$, e.g. `$0.0421$`).
  6. Calibration time in seconds in English math mode (e.g. `$45.2\text{ s}$`).
  7. Downstream SST-2 recovery accuracy in English math mode (e.g. `$89.40\%$`).

```latex
\chapter{پیوست ب: کاتالوگ جامع ۷۲ موتور فشرده‌سازی مدل}
\label{app:all_72_compressors}

\begin{longtable}{@{}cllcccr@{}}
\caption{کاتالوگ جامع ۷۲ موتور فشرده‌سازی ثبت‌شده در پلتفرم به تفکیک پارادایم و نسبت کاهش}
\label{tab:app_all_72_compressors} \\
\toprule
\textbf{ردیف} & \textbf{شناسه رجیستری} & \textbf{پارادایم فشرده‌سازی} & \textbf{ضریب کاهش} & \textbf{خطای بازسازی} & \textbf{زمان (s)} & \textbf{دقت بازیابی} \\
\midrule
\endfirsthead
\toprule
\textbf{ردیف} & \textbf{شناسه رجیستری} & \textbf{پارادایم فشرده‌سازی} & \textbf{ضریب کاهش} & \textbf{خطای بازسازی} & \textbf{زمان (s)} & \textbf{دقت بازیابی} \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{\textit{ادامه در صفحهٔ بعد...}} \\
\bottomrule
\endfoot
\bottomrule
\endlastfoot

۱  & \lr{\texttt{hnc\_svd\_compressor}}  & \lr{Low-Rank}    & $2.00\times$ & 0.0812 & 12.4 & 93.12\% \\
۲  & \lr{\texttt{awq\_quantizer}}        & \lr{Quantization}& $4.00\times$ & 0.0945 & 45.1 & 91.80\% \\
۳  & \lr{\texttt{wanda\_pruner}}         & \lr{Pruning}     & $2.00\times$ & 0.1120 & 8.2  & 88.45\% \\
۴  & \lr{\texttt{fw\_svd\_compressor}}   & \lr{Low-Rank}    & $2.00\times$ & 0.0890 & 14.1 & 92.15\% \\
\dots & \dots & \dots & \dots & \dots & \dots & \dots \\
۷۲ & \lr{\texttt{q\_svd\_hybrid}}        & \lr{Hybrid}      & $8.00\times$ & 0.1340 & 58.7 & 86.90\% \\
\end{longtable}
```

### Appendix C — Master CUDA C++ & Triton Kernel Execution Catalog
Target: `academic/thesis/chapter_all_modules_appendix.tex` (`app:all_kernels`).
- Document all 108 CUDA C++ and 26 Triton kernel packages with launch bounds, measured forward latency, speedup over vectorized PyTorch reference, and maximum divergence.

```latex
\chapter{پیوست ج: کارنامه اجرای ۱۰۸ کِرنل بومی \lr{CUDA C++} و ۲۶ ماژول \lr{Triton}}
\label{app:all_kernels}

\begin{longtable}{@{}cllcccc@{}}
\caption{مشخصات اجرای کِرنل‌های سفارشی پردازنده گرافیکی به همراه ابعاد پرتاب و شتاب محاسباتی}
\label{tab:app_all_kernels} \\
\toprule
\textbf{ردیف} & \textbf{نام کِرنل} & \textbf{سطح پیاده‌سازی} & \textbf{ابعاد بلاک} & \textbf{شتاب} & \textbf{حداکثر اختلاف} & \textbf{وضعیت} \\
\midrule
\endfirsthead
\midrule
\endhead
۱  & \lr{\texttt{selective\_scan\_fwd}} & \lr{Native CUDA C++} & $256 \times 1$ & 3.42$\times$ & $4.2 \times 10^{-4}$ & \lr{PASS} \\
۲  & \lr{\texttt{fused\_linear\_cross}} & \lr{Triton JIT}      & $128 \times 64$& 2.84$\times$ & $1.8 \times 10^{-3}$ & \lr{PASS} \\
\bottomrule
\end{longtable}
```

### Appendix D — 203 Individual Module Monographs
Target: `academic/thesis/attentions_latex/individual_modules/*.tex`.
- Maintain individual technical monographs for all 203 modules detailing mathematical formulation, forward algorithm pseudocode, memory access profile, and empirical Pareto curves.

---

## Persian Typography and XePersian Rules
Per `docs/thesis_agents/PERSIAN_WRITING_RULES.md`:
1. **Never use `\lr{}` in Math**: Use $\mathcal{O}(N d)$, not $\lr{\mathcal{O}(N d)}$.
2. **Never wrap `\item` in `\lr{}`**: Keep LaTeX structural commands outside `\lr{}`.
3. **Persian Digits in Tables**: Use Persian numerals (۱، ۲، ...، ۱۳۱) for table row counts.
4. **Mandatory ZWNJ (نیم‌فاصله)**: Use `U+200C` in technical terms (e.g. «مکانیزم‌های»، «پیوست‌های»، «فشرده‌سازها»).
5. **No early buggy runs**: Omit corrupted preliminary runs and undocumented model aliases.

---

## Verification Protocol
1. Confirm `academic/thesis/chapter_all_modules_appendix.tex` contains complete tables for all appendices.
2. Count rows in Table A.1 and Table B.1:
   ```bash
   # Check 131 rows in Appendix A
   grep -c "\\\\texttt{" academic/thesis/chapter_all_modules_appendix.tex
   ```
3. Verify that all 203 modules match the canonical registries in `neuralops/registry/`.
4. Ensure zero ellipsis (`\dots`) in place of actual data rows.
5. Check that all table columns conform to the XePersian `longtable` environment.

---

## Common Pitfalls to Avoid
- Truncating tables with ellipsis (`...`) instead of outputting all 131 or 72 rows.
- Placing `\lr{}` around math complexity expressions like $\mathcal{O}(N^2)$.
- Using English numerals for the Persian row counter column.
- Including non-canonical aliases or uncalibrated buggy runs.
- Omitting the `\endhead` and `\endfoot` repeated headers in multi-page `longtable`.
