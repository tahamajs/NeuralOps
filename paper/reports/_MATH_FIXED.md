# Phase 2: Mathematical Formulations Fixed

**Date:** 2026-09-20  
**Status:** COMPLETE  

This report details the mathematical errors identified and fixed across the theoretical sections of the A3 research paper.

---

## 1. Mathematical Fixes Inventory

1. **Equation (4.5) & (4.8) — Weight Sensitivity Scaling:**
   - *Previous error:* Stated $W_{\mathrm{weighted}} = W \cdot \mathrm{diag}(\sqrt{F} + \epsilon)$ or ambiguous scale interaction.
   - *Correction:* Formulated exact numerical scale $\mathbf{s} = \sqrt{\mathrm{clamp}(F, \min=10^{-8}) + \epsilon}$ with $\epsilon = 10^{-6}$ and RMS scale normalization $\mathbf{s}_{\mathrm{norm}} = \mathrm{clamp}(\frac{\mathbf{s}}{\|\mathbf{s}\|_2 / \sqrt{d_{\mathrm{in}}} + 10^{-8}}, 10^{-3}, 10^{3})$ in accordance with `a3_compressor.py:473-485`.

2. **Equation (4.10a) & (A.16) — Break-Even Bounds and Rank Ceiling:**
   - *Previous error:* Stated ceiling as $\min(\lfloor 0.5 \min(d_{\mathrm{in}}, d_{\mathrm{out}})\rfloor, 256)$.
   - *Correction:* Updated to $r_{\mathrm{ceiling}} = \min(\lfloor 0.90 \cdot r_{\mathrm{break\_even}}\rfloor, 320)$ where $r_{\mathrm{break\_even}} = \frac{d_{\mathrm{in}} d_{\mathrm{out}}}{d_{\mathrm{in}} + d_{\mathrm{out}}}$. For square $768 \times 768$ projections, $r_{\mathrm{break\_even}} = 384$, strictly guaranteeing $\ge 10\%$ parameter reduction on all factorized layers.

3. **Equation (4.11) — Adaptive Rank Escalation Formula:**
   - *Previous error:* Stepped toward $\min(d_{\mathrm{in}}, d_{\mathrm{out}})$ with floor step of 8.
   - *Correction:* Stepped directly toward $r_{\mathrm{ceiling}}$ with floor step 16:
     $$r_{k+1} = \min\!\left(r_{\mathrm{ceiling}},\, r_k + \max\!\left(16,\, \left\lfloor\frac{r_{\mathrm{ceiling}} - r_k}{\max(1, K - k)}\right\rfloor\right)\right)$$
     matching `a3_compressor.py:556`.

4. **Equation (4.13) & (A.11) — LayerNorm Affine Gain Recalibration:**
   - *Previous error:* Stated inverse ratio $\gamma \cdot \frac{\sigma_{\mathrm{current}}}{\sigma_{\mathrm{target}}}$, which would improperly amplify variance if signal expanded and suppress if attenuated.
   - *Correction:* Formulated compensation ratio $\gamma_{\mathrm{new}} = \gamma \odot \mathrm{clamp}(\frac{\sigma_{\mathrm{target}}}{\sigma_{\mathrm{current}}}, 0.5, 2.0)$ matching `pruning_utils.py:459`.

5. **Equation (3.7) — Outer Product vs Element-wise Squared Energy:**
   - *Previous error:* Displayed $F = \frac{1}{N} \sum x_t^2$ without vector notation.
   - *Correction:* Replaced with unambiguous Hadamard product notation $F = \frac{1}{N} \sum_{t=1}^N x_t \odot x_t$.

6. **Appendix A.4 — Centering Shift and Affine Compensation:**
   - *Previous error:* Retained unreferenced scalar bias equation (A.12).
   - *Correction:* Streamlined Appendix A.4 to focus on the closed-form dynamic range gain compensation with explicit $[0.5, 2.0]$ numerical stability clamping.

7. **Table 8 (tab:parameter_counting_formulation) & Table 14 (tab:app_notation_summary):**
   - *Previous error:* Stated $N=1000$ and $r_{\mathrm{ceiling}} \le 256$ unconditionally.
   - *Correction:* Reconciled parameters across both tables to reflect calibrated $N=500$, $r_{\mathrm{ceiling}} = \min(\lfloor 0.9 r_{\mathrm{break\_even}}\rfloor, 320)$, and $\tau \in \{0.40, 0.50\}$.

---

## 2. Compilation Verification
Document re-compiled with 0 LaTeX math errors and consistent numbering across equations and tables.
