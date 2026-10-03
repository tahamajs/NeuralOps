# A3 Mathematical and Formula Audit Report

**Date:** September 18, 2026  
**Artifact:** `paper/A3/main.pdf` (Symlinked to `paper/PerModulePaper/a3/main.pdf`)  
**Status:** PASS (0 errors, 30 pages compiled cleanly)  

---

## Executive Summary
All 10 mathematical, dimensional, and formula issues across `paper/A3/` have been verified and rectified. Derivations in the main text and Appendix A are now completely aligned, dimensionally consistent, and mathematically rigorous.

---

## Detailed Audit of 10 Mathematical Fixes

### Fix 1 — LayerNorm Equation (3.9) Standard Deviation Definition
- **File:** `paper/PerModulePaper/a3/03_background.tex`
- **Before:**
  $$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta, \quad \sigma = \mathbb{E}[(x-\mu)^2] + \epsilon \quad \text{(Variance labeled as std)}$$
- **After:**
  $$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\mathrm{Var}(x) + \epsilon}} + \beta, \quad \text{where } \mathrm{Var}(x) = \mathbb{E}[(x - \mu)^2], \quad \sigma = \sqrt{\mathrm{Var}(x) + \epsilon}$$
- **Dimensional Verification:** $\frac{x - \mu}{\sigma}$ is dimensionless; multiplied by scale vector $\gamma \in \mathbb{R}^{d_\text{model}}$ and shifted by $\beta \in \mathbb{R}^{d_\text{model}}$, yielding proper dimension $\mathbb{R}^{d_\text{model}}$. Matches Appendix A.4 Equation (A.10).

---

### Fix 2 — Truncated SVD Frobenius Norm Formulation (3.2)
- **File:** `paper/PerModulePaper/a3/03_background.tex`
- **Before:**
  $$\text{"with error } \|W - W_r\|_F = \sum_{i > r} \sigma_i^2\text{"} \quad \text{(Squared norm missing power)}$$
- **After:**
  $$\text{"with squared error } \|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2, \text{ or equivalently } \|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}\text{"}$$
- **Dimensional Verification:** $\|W - W_r\|_F$ has units of matrix entries; $\sum_{i > r} \sigma_i^2$ has squared units. The square root resolves the dimensional mismatch.

---

### Fix 3 — Weighted SVD Input Transformation (4.5)
- **File:** `paper/PerModulePaper/a3/04_method.tex`
- **Before:**
  $$W_\text{weighted} = W \cdot \mathrm{diag}(\sqrt{F} + \epsilon) \quad \text{($\epsilon$ outside root)}$$
- **After:**
  $$W_\text{weighted} = W \cdot \mathrm{diag}(\sqrt{F + \epsilon})$$
- **Dimensional Verification:** $F_i \approx \mathbb{E}[x_i^2]$ has squared activation units. Stability constant $\epsilon = 10^{-6}$ is added inside the square root prior to scaling $W$, matching the unweighting operation $V_r^\text{unweighted} = V_r^\top \cdot \mathrm{diag}(1/\sqrt{F + \epsilon})$.

---

### Fix 4 — LayerNorm $\gamma$ Scale Adjustment (4.13) & Algorithm 1
- **File:** `paper/PerModulePaper/a3/04_method.tex`
- **Before:**
  $$\gamma_\text{new} = \gamma \cdot \frac{\sigma_\text{target}}{\sigma_\text{current}} \quad \text{(Inverted)}$$
- **After:**
  $$\gamma_\text{new} = \gamma \cdot \frac{\sigma_\text{current}}{\sigma_\text{target}}$$
  *(Also corrected in Algorithm 1 line 23).*
- **Mathematical Derivation:**
  $$y_\text{pre} = \gamma \odot \frac{x - \mu}{\sigma_\text{target}}, \quad y_\text{post} = \gamma_\text{new} \odot \frac{x - \mu}{\sigma_\text{current}}$$
  $$\text{Equating representations } y_\text{post} = y_\text{pre} \implies \frac{\gamma_\text{new}}{\sigma_\text{current}} = \frac{\gamma}{\sigma_\text{target}} \implies \gamma_\text{new} = \gamma \odot \frac{\sigma_\text{current}}{\sigma_\text{target}}.$$
  This matches Appendix A.4 Equation (A.11) ($\gamma_\text{new} = \gamma \odot \frac{\sigma_\text{post}}{\sigma_\text{pre}}$ where $\sigma_\text{post} = \sigma_\text{current}$ and $\sigma_\text{pre} = \sigma_\text{target}$).

---

### Fix 5 — Empirical Fisher Proxy Caveat (3.6)
- **File:** `paper/PerModulePaper/a3/03_background.tex`
- **Text Added:**
  > "This is a heuristic approximation under the assumptions that (i) the downstream loss has roughly uniform sensitivity across output dimensions, and (ii) cross-channel correlations are negligible. It is not an exact identity; we use it as a cheap proxy for the true Fisher diagonal."
- **Verification:** Accurately grounds the diagonal activation energy proxy $F_i \approx \mathbb{E}[x_i^2]$ without asserting exact mathematical equivalence to the true expected Hessian diagonal.

---

### Fix 6 — Appendix A.4 LayerNorm Centering Shift ($\beta$ Update)
- **File:** `paper/PerModulePaper/a3/A_math.tex`
- **Formulation:**
  $$\beta_{\mathrm{new}} = \beta + \gamma_{\mathrm{new}} \odot \frac{\mu_{\mathrm{post}} - \mu_{\mathrm{pre}}}{\sigma_{\mathrm{post}}} = \beta + \gamma \odot \frac{\mu_{\mathrm{post}} - \mu_{\mathrm{pre}}}{\sigma_{\mathrm{pre}}}$$
- **Mathematical Derivation:**
  $$\beta_\text{new} - \gamma_\text{new} \odot \frac{\mu_\text{post}}{\sigma_\text{post}} = \beta - \gamma \odot \frac{\mu_\text{pre}}{\sigma_\text{pre}}$$
  $$\text{Substituting } \frac{\gamma_\text{new}}{\sigma_\text{post}} = \frac{\gamma}{\sigma_\text{pre}} \implies \beta_\text{new} = \beta + \gamma \odot \frac{\mu_\text{post} - \mu_\text{pre}}{\sigma_\text{pre}}.$$
  The sign is rigorously verified as positive for $(\mu_\text{post} - \mu_\text{pre})$.

---

### Fix 7 — Non-Trivial Upper Bound on Factorization Error (4.14)
- **File:** `paper/PerModulePaper/a3/04_method.tex`
- **Text Added:**
  Combined with the weighted SVD tail bound from Appendix A.3:
  $$\|W - U_r V_r\|_F \leq \frac{\sqrt{\sum_{i > r} \sigma_i^2(W_\text{weighted})}}{\sqrt{\min_j F_j}}$$
- **Mathematical Verification:** Proves that total Frobenius reconstruction error in original weight space is strictly controlled by tail singular values of $W_\text{weighted}$ divided by the minimal channel sensitivity $\min_j F_j$.

---

### Fix 8 — Notation Harmonization: Section 4.5 vs Appendix A.4
- **Verification:**
  - Section 4.5 Equation (4.13): $\gamma_\text{new} = \gamma \cdot \frac{\sigma_\text{current}}{\sigma_\text{target}}$
  - Appendix A.4 Equation (A.11): $\gamma_\text{new} = \gamma \odot \frac{\sigma_\text{post}}{\sigma_\text{pre}}$
  - Identity confirmed: $\sigma_\text{target} \equiv \sigma_\text{pre}$ and $\sigma_\text{current} \equiv \sigma_\text{post}$.

---

### Fix 9 — Table 2 (Notation Summary) Dimensional Precision
- **File:** `paper/PerModulePaper/a3/03_background.tex`
- **Formulation:**
  $\Sigma$ dimension set to $r \times r$ (for truncated singular value diagonal matrix). All 19 symbols aligned with standard typography and dimension declarations.

---

### Fix 10 — Consistency in Quantities and Metrics
- **Files:** `paper/PerModulePaper/a3/09_discussion.tex`, `paper/PerModulePaper/a3/01_intro.tex`, `paper/PerModulePaper/a3/main.tex`
- **Verification:**
  - Discarded all obsolete legacy placeholders ($5.0\times$, $13.5$M parameters, $88.5\%$).
  - Section 9.1.1: Verified at $1.02\times$ parameter reduction.
  - Section 9.4: Baseline $66.95$M $\to 65.71$M params ($1.02\times$ reduction), Disk $134$ MB $\to 131.5$ MB ($1.02\times$), Accuracy $90.71\% \to 90.02\%$ ($\Delta -0.69\%$).
  - Section 9.5: $\textbf{A3 (r=128)}$ reported at $1.02\times$ and $90.02\%$.
  - Section 1.1: Completed sentence on channel-wise variance without dangling fragments.

---

## Phase 2 Verification Checklist

| Verification Step | Target Criteria | Actual Status | Verdict |
| :--- | :--- | :--- | :--- |
| **Check 1: Dimensionality** | All equations dimensionally consistent | Fully verified across (3.2), (3.9), (4.5), (4.13), (A.10)-(A.12) | **PASS** |
| **Check 2: Section $\leftrightarrow$ Appendix** | Equations in text match derivations | Eq (4.13) $\equiv$ Eq (A.11); Eq (4.14) $\equiv$ Eq (A.8) | **PASS** |
| **Check 3: Compilation** | `pdflatex main.tex` 0 errors | 2 passes cleanly compiled; exactly 30 pages | **PASS** |
| **Check 4: Report Generated** | [A3_MATH_FIX_REPORT.md](audit/A3_MATH_FIX_REPORT.md) | Created with comprehensive documentation | **PASS** |
