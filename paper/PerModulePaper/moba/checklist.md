# Pre-Submission Checklist — Mamba SSM

## Setup
- [ ] Copied `_TEMPLATE/` to `mamba/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module mamba`
- [ ] No `[PLACEHOLDER]` remains

## Critical Honesty Checks
- [ ] **Docstring O(L) claim is contradicted in paper** — paper says O(L²d)
- [ ] **Module is described as SSM-inspired, not true Mamba**
- [ ] **Quadratic complexity is explicit**
- [ ] **Weight reuse is heuristic, acknowledged**

## Abstract
- [ ] Under 250 words
- [ ] Honest about complexity
- [ ] SSM-inspired, not Mamba
- [ ] Preservation gate mentioned

## Method
- [ ] Decay formula
- [ ] Two-branch projection
- [ ] Preservation gate
- [ ] **Complexity $\mathcal{O}(L^2 d)$ stated**

## Results
- [ ] Main results (Table 7.1)
- [ ] Preservation gate ablation
- [ ] Decay ablation
- [ ] Scaling table (quadratic visible)

## Limitations
- [ ] O(L²) stated
- [ ] Not true Mamba
- [ ] Weight reuse heuristic

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/mamba
nano papers/mamba/variables.json
cd papers/mamba
python apply_vars.py --module mamba
grep -rn "{{" .
python build.py --module mamba --pdf
```
```

---

## 🎯 خلاصه نهایی

### هشدارهای حیاتی برای مقاله

1. **docstring O(L) غلط است** — در paper بگو O(L²d)
2. **این Mamba واقعی نیست** — SSM-inspired
3. **Weight reuse heuristic است** — نه principled
4. **Preservation gate نوآوری اصلی است**
5. **`copy_weights` هنوز `raise` می‌کند** — fix شود

### نوآوری‌های قابل برجسته‌سازی

1. **Two-branch formulation** — Q/K/V reuse
2. **Learned per-dim decay** — $\gamma_i = \sigma(-\exp(A_{\log, i}))$
3. **Preservation gate** — stability
4. **Causal + bidirectional** modes

### مقایسه با مقالات قبلی

| مقاله | Complexity | Parameter |
|---|---|---|
| A3/AASVD | $\mathcal{O}(L^2)$ | کاهش |
| Compressed MHA | $\mathcal{O}(L^2)$ | کاهش |
| Linformer | $\mathcal{O}(Lk)$ | افزایش |
| TCA | $\mathcal{O}(Lk)$ | افزایش |
| **Mamba SSM** | **$\mathcal{O}(L^2 d)$** | **کاهش جزئی** |

### نکات کلیدی در `variables.json`

- `LORA_RANK = 16`
- `D_PARAMS = 52.0M` (کاهش ~22% از 66.9M)
- `D_ACC = 88.0%` (افت ~2.7%)

---

**قدم بعدی: `run_all_4_mamba.py` را اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار. یا اگر می‌خواهی fix `copy_weights` را انجام بدهم، بگو.**