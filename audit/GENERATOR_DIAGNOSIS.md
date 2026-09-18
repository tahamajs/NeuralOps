# Root Cause Diagnosis — PerModulePaper Generator Word Count Deficit

## 1. Executive Summary
The generator was stuck in a loop recompiling Batch FIX-0 because while `enhanced_sections.py` was heavily expanded in **LaTeX (`.tex`)** (producing 7,493 TeX words, 15 compiled PDF pages, and ~485–510 KB), the corresponding **Markdown (`.md`)** files were left as concise skeletons in many sections (totaling ~5,069–5,816 words). 

Because `verify_module_paper.py` evaluates word counts exclusively across the 19 Markdown (`.md`) files (`total_words += len(clean.split())`), it emitted warnings (`Total word count 5535 < 6000`), while the PDF generator achieved 15 pages due to LaTeX tabular structures, lstlisting blocks, and derivations.

---

## 2. Root Cause Analysis

### 2.1 Discrepancy Between Markdown and LaTeX Content
In `scripts/enhanced_sections.py`:
- Line 1282: `md["B_implementation.md"]` contains only **31 words**, while `tex["B_implementation.tex"]` has **370 words** with 4 code listings.
- Line 1378: `md["C_data.md"]` contains only **47 words**, while `tex["C_data.tex"]` has **425 words** with 3 comprehensive empirical tables.
- Line 1459: `md["D_negative_results.md"]` contains only **37 words**, while `tex["D_negative_results.tex"]` has **258 words** with 4 failure analyses.
- Line 1477: `md["E_ethics.md"]` contains only **37 words**, while `tex["E_ethics.tex"]` has **647 words** with 6 subsections and an energy breakdown table.
- Line 1193: `md["A_math.md"]` contains **116 words**, while `tex["A_math.tex"]` has **610 words** with Theorems 3 & 4.
- Line 952: `md["08_analysis.md"]` contains **240 words**, while `tex["08_analysis.tex"]` has **660 words** with layer sensitivity and ablation tables.

### 2.2 Section Word-Count Deficit Breakdown
Comparing actual current Markdown word count against the master template targets:

| Section | Current MD Words | Target Words | Deficit | Status |
|---|---|---|---|---|
| `00_meta.md` | 239 | 200 | +39 | ✅ Meets target |
| `01_intro.md` | 451 | 500 | -49 | ⚠️ Near target (90%) |
| `02_related.md` | 294 | 500 | -206 | ❌ Below 60% |
| `03_background.md` | 464 | 600 | -136 | ⚠️ Near target (77%) |
| `04_method.md` | 651 | 1,000 | -349 | ❌ Below 70% |
| `05_architecture.md` | 499 | 600 | -101 | ⚠️ Near target (83%) |
| `06_setup.md` | 240 | 500 | -260 | ❌ Below 50% |
| `07_results.md` | 1,518 (with CSV) | 1,000 | +518 | ✅ Meets target |
| `08_analysis.md` | 240 | 800 | -560 | ❌ Below 30% |
| `09_discussion.md` | 390 | 600 | -210 | ❌ Below 65% |
| `10_limitations.md` | 215 | 400 | -185 | ❌ Below 55% |
| `11_conclusion.md` | 120 | 300 | -180 | ❌ Below 40% |
| `12_reproducibility.md`| 70 | 200 | -130 | ❌ Below 35% |
| `13_references.md` | 157 | 200 | -43 | ⚠️ Near target (78%) |
| `A_math.md` | 116 | 400 | -284 | ❌ Below 30% |
| `B_implementation.md` | 31 | 400 | -369 | ❌ Below 10% |
| `C_data.md` | 47 | 300 | -253 | ❌ Below 20% |
| `D_negative_results.md`| 37 | 300 | -263 | ❌ Below 15% |
| `E_ethics.md` | 37 | 200 | -163 | ❌ Below 20% |
| **TOTAL** | **5,816** | **8,000** | **-2,184** | **Deficit: 27.3%** |

### 2.3 Shortest Sections (< 50% of Target)
The 8 sections with severe deficits are:
1. `B_implementation.md`: 31 words (target 400, deficit -369, 7.8% of target)
2. `D_negative_results.md`: 37 words (target 300, deficit -263, 12.3% of target)
3. `C_data.md`: 47 words (target 300, deficit -253, 15.7% of target)
4. `E_ethics.md`: 37 words (target 200, deficit -163, 18.5% of target)
5. `A_math.md`: 116 words (target 400, deficit -284, 29.0% of target)
6. `08_analysis.md`: 240 words (target 800, deficit -560, 30.0% of target)
7. `12_reproducibility.md`: 70 words (target 200, deficit -130, 35.0% of target)
8. `11_conclusion.md`: 120 words (target 300, deficit -180, 40.0% of target)
9. `06_setup.md`: 240 words (target 500, deficit -260, 48.0% of target)

---

## 3. Analysis of Code Generation Architecture
- **Hard-coded vs. Dynamic**:
  - `07_results.md` is fully dynamic, pulling exact rows from mined benchmark CSV logs with line-level citations.
  - `00_meta.md` and `B_implementation.md` dynamically inspect the module file, line counts, class names, and init arguments.
  - `04_method.md`, `05_architecture.md`, `A_math.md`, and appendices have rigorous mathematical models and derivations, but currently exist in full detail only in LaTeX (`.tex`), leaving Markdown counterparts as stubs.
- **Feasibility of 8,000 Words**:
  - If the rich LaTeX explanations, algorithms, derivations, and hardware profiling details already present in `enhanced_sections.py` are mapped equivalently into their respective Markdown sections, the total Markdown word count immediately rises from **5,816 words** to **~8,200 words** without fabricating numbers or repeating sentences.

---

## 4. Required Changes in `enhanced_sections.py`
1. **Mirror LaTeX content into Markdown**:
   - `A_math.md`: Add complete derivations of Theorem 1 (energy truncation bound), Theorem 2 (subspace projection stability), Theorem 3 (Mirsky's perturbation bound), and Theorem 4 (dynamical stability).
   - `B_implementation.md`: Include code snippets for `StructuredModuleWrapper`, `GPUMonitor`, and `DispatchEngine`.
   - `C_data.md`: Include the complete validation split markdown tables, step-latency breakdowns, and singular decay profiles.
   - `D_negative_results.md`: Detail the 4 failure modes (uncalibrated truncation, high LoRA learning rates, asymmetric singular splitting, and long-context saturation).
   - `E_ethics.md`: Add the datacenter power accounting, edge deployment thermal modeling, and energy dissipation tables.
   - `08_analysis.md`: Mirror the layer sensitivity table, $r_L$ adapter ablation table, and gradient flow text.
   - `06_setup.md`: Expand dataset token splits, optimizer hyperparameters, and hardware profiling specifications.
   - `02_related.md`: Expand connections to SVD, low-rank attention, and 2:4 structured pruning literature.
