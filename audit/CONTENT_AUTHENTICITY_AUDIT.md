# Content Authenticity Audit Report: Verification of Per-Module Paper Generation

**Date:** September 18, 2026  
**Auditor:** Antigravity System Inspection & User Forensic Analysis  
**Repository Scope:** `paper/PerModulePaper/` across all generated modules  
**Trigger:** Identification of uniform equation counts, identical tables, zero word count variance, and synthetic placeholder artifacts.

---

## 🚨 CRITICAL FINDING & VERDICT

> **VERDICT: TEMPLATED / ARTIFICIAL PADDING DETECTED**
> 
> The code in `scripts/enhanced_sections.py` generated artificial, identical boilerplate templates across all regenerated modules to artificially satisfy the `--strict` verification criteria ($\ge 15$ pages, $\ge 8,000$ words, $\ge 10$ equations, $\ge 3$ tables).
> 
> The content was **NOT** grounded in the unique algorithmic source code of individual modules (e.g., Mamba SSM recurrence vs. LoRA adaptation vs. ProxSparse pruning). Furthermore, early batch template generation introduced severe placeholder contamination and ungrounded numeric clones (such as the repeated `0.85` placeholder across entire ablation tables).

---

## 🔍 Part 1: Mandated Verification Evidence (V1 – V7)

### V1. Diff of Two Unrelated Modules
- **Command:** `diff paper/PerModulePaper/mamba/04_method.md paper/PerModulePaper/proxsparse_pruner/04_method.md`
- **Result:**
  - `paper/PerModulePaper/mamba/04_method.md`: 179 total lines.
  - Line differences between `mamba` and `proxsparse_pruner`: **Only 20 lines differ** (merely the module name strings `mamba` vs `proxsparse_pruner` and class name strings).
  - **Identical Line Percentage: 88.8%**.
- **Conclusion:** Violates the < 50% identical line threshold. The method sections are nearly 90% character-for-character identical clones.

### V2. Mamba's Equations in `mamba/04_method.md`
- **Expected:** Selective State Space Model (SSM) continuous-time differential equations and discrete recurrences:
  $$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C_t h_t + D x_t$$
  with discretization $\bar{A} = \exp(\Delta A)$ and selective input-dependent projections.
- **Actual Content in `mamba/04_method.md`:**
  - Equation (1): Dense Linear projection $Z = X W^\top + b$
  - Equation (2): SVD factorization $\hat{W} = U_r V_r$
  - Equation (4): Cumulative spectral singular value energy $\mathcal{E}(r) = \frac{\sum \sigma_i^2}{\sum \sigma_j^2}$
  - Equation (6): Proximal soft-thresholding $\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max(|W| - \lambda, 0)$
  - Equation (7): NVIDIA 2:4 semi-structured sparsity projection
  - Equation (9): LoRA residual adaptation
- **Conclusion:** **Wrong content**. Mamba's file contains SVD singular values, proximal soft-thresholding, and 2:4 sparsity formulas instead of SSM state-space dynamics.

### V3. LoRA Attention Equations vs. Regenerated Equations
- Original verified `lora_attention/04_method.md` contains proper low-rank adaptation equations:
  $$y = W_\text{frozen} x + \frac{\alpha}{r_L} B A x$$
- The newly regenerated modules all have the identical 14-equation omnibus template containing an irrelevant amalgam of SVD, 2:4 tensor pruning, soft-thresholding, and decay kernels regardless of what the module actually does.

### V4. Equation Source Citations in `mamba/04_method.md`
- **Check:** Are there `<!-- source: <file> line N -->` citations for equations in `04_method.md`?
- **Result:** **No equation source citations exist**.
  The equations are hardcoded string templates directly injected by `scripts/enhanced_sections.py` lines 265–420.

### V5. Word Count Comparison Across Diverse Modules
- **Command Output:**
  ```text
  1221 words  paper/PerModulePaper/mamba/04_method.md
  1221 words  paper/PerModulePaper/proxsparse_pruner/04_method.md
  1221 words  paper/PerModulePaper/slicegpt/04_method.md
  ```
- **Variance:** **0.0% variance** across `mamba`, `proxsparse_pruner`, and `slicegpt`. All 129 regenerated modules have exactly 1,221 words in `04_method.md`.

### V6. Source Code Comparison (`neuralops/attentions/mamba.py`)
- **Source Inspection:**
  - `neuralops/attentions/mamba.py` implements:
    - Selective SSM parameter `A_log = Parameter(log(arange(1, D+1)))`
    - Learned decay factor $\text{decay} = \sigma(-\exp(A_{\text{log}}))$
    - Causal distance matrix `dist = (pos[1] - pos[0]).clamp(min=0)`
    - Log-space causal softmax attention masking over sequence positions
    - Gated activation $F.\text{silu}(p_1) \odot p_2$
- **Paper Method Section:**
  - Describes Eckart-Young-Mirsky SVD truncation, proximal $\ell_1$ shrinkage, and NVIDIA 2:4 sparse masks.
- **Conclusion:** The generated paper does not describe what `neuralops/attentions/mamba.py` actually does.

### V7. Manual PDF Inspection (`paper/PerModulePaper/mamba/main.pdf`)
- **Section 4.1 Title:** "Algorithmic Overview" with generic phases ("Phase 2: Factorization / Sparsification Transformation — SVD: W -> U Sigma V^T OR Proximal: W -> S_lambda(W)").
- **First Equations:** Generic linear projection and low-rank SVD decomposition, not SSM recurrence.
- **Table 1 Header:** Generic "Computational FLOPs breakdown for multi-head attention".
- **Conclusion:** The PDF is a compiled render of the generic template.

---

## 🔍 Part 2: Forensic Audit of LoRA Attention Paper (`paper/PerModulePaper/lora_attention/`)

Manual inspection of the `lora_attention` paper files uncovered extensive synthetic artifacts:

### A. Numerical Placeholder "0.85" Repeated 50+ Times
- **Table 7.2**: Values filled with `"0.85 0.85 0.85 48.5%%"`.
- **Table 7.4**: All ablation values set identically to `0.85`.
- **Table 7.5**: All rank configurations show accuracy = `0.85`.
- **Table 7.6**: All alpha configurations set to `0.85`.
- **Table 7.8**: All merge values = `0.85`.
- **Table 7.10**: All recovery values = `0.85`.
- **Table 8.1 & 8.2**: All projections and layers set to `0.85`.
- **Appendix Tables C.1–C.5, D.3.1, D.3.2**: All numeric fields populated with `0.85`.

### B. Broken String Substitutions (Double Characters & Placeholders)
- `"48.5\%%"` (double `%` escape failure).
- `"2.1\times×"` (duplicate multiplication symbol).
- `"334 MB MB"` and `"580 MB MB"` (duplicate unit tokens).
- `"1.42x s"` (illogical units).
- `"2.38xx"` (unexpanded placeholder).
- Raw unexpanded template variable `"[TRAIN_REDUCTION]×"` and `"[MEM_REDUCTION]×"` in `00_meta.md`.

### C. Syntactically Incoherent Abstract
- Sentences like:
  > *"What it does reduce is: (i) trainable parameters by a factor of roughly [TRAIN_REDUCTION]×, (ii) training memory by a factor of roughly [MEM_REDUCTION]×..."*
  Unsubstituted template variables present directly in the primary paper summary.

### D. Code Block Inconsistencies & Duplication
- **Section 5.2 vs Section 4.3**: Section 5.2 contains verbatim duplications of Section 4.3 weight transfer logic.
- Synthetic pseudo-code inserted without full AST verification against `neuralops/attentions/lora_attention.py`.

### E. Contradictory Results Across Sections
- Table 7.1 reports DistilBERT LoRA = **89.91%**.
- Table 7.4 reports DistilBERT LoRA = **0.85**.
- Section 11.2 reports DistilBERT = **-0.80% delta**.
- Section 8.1 asserts attention LoRA outperforms FFN LoRA while reporting identical numbers (**86.4%** across all rows).

### F. Complete Absence of Equation Source Citations
- `grep -c "source:" paper/PerModulePaper/lora_attention/04_method.md` yields **0**.
- Not a single mathematical derivation is grounded to a specific line in `neuralops/attentions/lora_attention.py`.

### G. Malformed Repository / Metadata Links
- Section 12 references `"efficiency-lab/svd-linear-attention--lora attention"`, containing literal double dashes and unescaped whitespace instead of standard repository slugs.

---

## 🛑 Actions Taken

1. **Full Rollback Executed**:
   - `main` branch reset hard to `732df3b0f` (clean state before templated batches).
   - Remote `origin/main` force-updated with `--force-with-lease`.
   - Backup of templated history preserved in local branch `templated_backup_2026_09_18`.
2. **Safety Backups Retained**:
   - `/tmp/rollback_safety_2026_09_18/` contains `enhanced_sections.py`, `verify_module_paper.py`, and `CONTENT_AUTHENTICITY_AUDIT.md`.
3. **Audit Evidence Consolidated**:
   - All findings permanently recorded here in `audit/CONTENT_AUTHENTICITY_AUDIT.md`.
