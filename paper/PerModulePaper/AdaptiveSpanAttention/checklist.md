# Pre-Submission Checklist — Adaptive Span Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_span/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_span`
- [ ] No `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] Soft ramp formula stated
- [ ] L1 span penalty explained
- [ ] Learned span results shown

## Method
- [ ] Ramp formula numbered
- [ ] Span logit formula numbered
- [ ] Fast path optimization mentioned
- [ ] Complexity unchanged (stated honestly)

## Results
- [ ] Main table
- [ ] Learned span distribution
- [ ] Per-layer span
- [ ] Max span ablation
- [ ] Ramp size ablation
- [ ] L1 penalty ablation

## Honesty
- [ ] No parameter reduction — stated
- [ ] No speedup without hard truncation — stated
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation

## Interpretability
- [ ] Bimodal span distribution shown
- [ ] Layer-wise pattern shown
- [ ] Comparison with fixed window

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final
- [ ] Read aloud
- [ ] PDF compiles
- [ ] External feedback

---

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/adaptive_span
nano papers/adaptive_span/variables.json
cd papers/adaptive_span
python apply_vars.py --module adaptive_span
grep -rn "{{" .  # verify empty
python build.py --module adaptive_span --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_span
```

### مرحله ۲: اجرای ASA-Span

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_span.py --models distilbert,roberta,gpt2 --max_span 128 --ramp_size 32 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adaptive_span
python apply_vars.py --module adaptive_span
python build.py --module adaptive_span
```

---

## 💡 نکات کلیدی ASA-Span

### ۱. تفاوت اصلی با ASA-Sparse

| جنبه | ASA-Sparse | ASA-Span |
|---|---|---|
| انتخاب | top-k (content-aware) | distance-based |
| پارامتر | sparsity_logit | span_logit |
| Mask | sigmoid soft top-k | linear ramp |
| Regularization | ندارد | L1 روی span |

### ۲. Soft Ramp — مزیت اصلی

$$
g_z(d) = \text{clamp}\left(\frac{R + z - d}{R}, 0, 1\right)
$$

- **Piecewise linear** → ساده
- **Differentiable** → قابل یادگیری
- **No hard cutoff** → بدون artifact مرزی

### ۳. Initialize Open, Learn to Shrink

- Span اولیه = $S_\text{max}$ (کاملاً باز)
- L1 penalty → باعث shrink تدریجی
- Equilibrium: span به اندازه‌ای که accuracy حفظ شود

### ۴. Head Heterogeneity — مهم‌ترین تفسیر

از روی learned spans می‌فهمی:
- کدام heads **local** هستند (span ~16)
- کدام heads **global** هستند (span ~128)
- این **ساختار** در روش‌های دیگر دیده نمی‌شود

### ۵. بدون Speedup

ASA-Span **سرعت inference را کم نمی‌کند** چون همه scores محاسبه می‌شوند. برای speedup واقعی:

1. Train با soft ramp
2. Extract learned spans
3. Hard truncate at inference
4. Sparse kernel

این محدودیت را **صادقانه** در abstract و discussion ذکر کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_adaptive_span/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**