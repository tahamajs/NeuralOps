# Phase 1: Code-Paper Alignment Report

**Date:** 2026-09-20  
**Status:** COMPLETE  

This report documents the reconciliation of the 8 discrepancies between the A3 code (`Codes/framework/compression_engine/a3_compressor.py`, `Codes/framework/compression_engine/pruning_utils.py`) and the A3 research paper (`paper/PerModulePaper/a3/`).

---

## Reconciliation Summary Table

| Parameter / Formulation | Paper Original | Code Implementation | Reconciled Paper Status | Source Reference |
|:---|:---|:---|:---|:---|
| **1. Quality Threshold ($\tau$)** | $\tau = 0.40$ strictly | `max_error = 0.50` default in `A3Compressor.__init__` | Default elevated to $\tau = 0.50$ (unlocking Pareto point); $\tau = 0.40$ preserved as conservative baseline | `a3_compressor.py:243`, `04_method.tex`, `06_setup.tex` |
| **2. Target Rank ($r$)** | Fixed $r = 128$ | `rank = 64` default in `A3Compressor.__init__` | Clarified as $r = 64$ default / candidate; $r=128$ evaluated in sweep comparisons | `a3_compressor.py:239`, `06_setup.tex` |
| **3. Calibration Size ($N$)** | $N = 1000$ (Sec 6.2) vs. $500$ (Sec 9.3) | `n_calib = 500` default in `A3Compressor.__init__` | Reconciled consistently across paper to $N = 500$ sequences | `a3_compressor.py:250`, `06_setup.tex`, `A_math.tex` |
| **4. Rank Ceiling ($r_{\mathrm{ceiling}}$)** | $\min(\lfloor 0.5 \min(d_{\mathrm{in}}, d_{\mathrm{out}})\rfloor, 256)$ | $\min(\lfloor 0.90 \cdot r_{\mathrm{break\_even}}\rfloor, 320)$ | Reconciled to exact code formula $\min(\lfloor 0.90 \cdot r_{\mathrm{break\_even}}\rfloor, 320)$ | `a3_compressor.py:521`, `04_method.tex`, `A_math.tex` |
| **5. LoRA Rank ($r_{\mathrm{LoRA}}$)** | Unconditionally instantiated at $r_L = 16$ or $8$ | `lora_rank = 0` default on plain linear layers | Clarified that plain linear layers without pre-existing LoRA do not attach extra LoRA weights | `a3_compressor.py:591`, `06_setup.tex`, `A_math.tex` |
| **6. Rank Escalation Step** | $\max(8, \lfloor\frac{\min(d_{\mathrm{in}}, d_{\mathrm{out}}) - r_k}{K - k}\rfloor)$ | $\max(16, \lfloor\frac{r_{\mathrm{ceiling}} - r_k}{\max(1, K - k)}\rfloor)$ | Reconciled equation 4.11 to match code escalation toward $r_{\mathrm{ceiling}}$ | `a3_compressor.py:556`, `04_method.tex` |
| **7. LayerNorm Recalibration** | Ratio $\gamma \cdot \frac{\sigma_{\mathrm{current}}}{\sigma_{\mathrm{target}}}$ | Correction $\gamma \odot \mathrm{clamp}(\frac{\sigma_{\mathrm{target}}}{\sigma_{\mathrm{current}}}, 0.5, 2.0)$ | Corrected formulation to inverse scale compensation $\frac{\sigma_{\mathrm{target}}}{\sigma_{\mathrm{current}}}$ with $[0.5, 2.0]$ clamping | `pruning_utils.py:459`, `04_method.tex`, `A_math.tex` |
| **8. Weight Sensitivity Scaling** | $W \cdot \mathrm{diag}(\sqrt{F} + \epsilon)$ | $\mathrm{scale} = \sqrt{\mathrm{clamp}(F, \min=10^{-8}) + 10^{-6}}$ | Fixed equation 4.8 and 4.9 to match code root clamping and RMS scale normalization | `a3_compressor.py:473`, `04_method.tex` |

---

## Verification Evidence
All modified `.tex` files compile cleanly without errors under `pdflatex` in `paper/PerModulePaper/a3/`.
