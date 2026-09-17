# Pre-Submission Checklist — HALO-ReGLA

## Setup
- [ ] Copied `_TEMPLATE/` to `halo_regla/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module halo_regla`
- [ ] No `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] Dense + linear streams described
- [ ] Per-head gate described
- [ ] Gate initialization explained

## Method
- [ ] Dense stream formula numbered
- [ ] Linear stream formula numbered
- [ ] Gate formula numbered
- [ ] Prefix-sum causality proven

## Results
- [ ] Main table
- [ ] Gate distribution
- [ ] Per-layer gate
- [ ] Gate bias ablation
- [ ] Kernel ablation
- [ ] LoRA rank ablation

## Honesty
- [ ] No parameter reduction — stated
- [ ] No speedup without hard routing — stated
- [ ] Failure cases in Appendix D
- [ ] Cumsum memory limitation stated

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
cp -r papers/_TEMPLATE papers/halo_regla
nano papers/halo_regla/variables.json
cd papers/halo_regla
python apply_vars.py --module halo_regla
grep -rn "{{" .  # verify empty
python build.py --module halo_regla --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE halo_regla
```

### مرحله ۲: اجرای HALO-ReGLA

```powershell
cd D:\SVD_Project
python run_all_4_halo_regla.py --models distilbert,roberta,gpt2 --gate_init_bias 1.8 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\halo_regla
python apply_vars.py --module halo_regla
python build.py --module halo_regla
```

---

## 💡 نکات کلیدی HALO-ReGLA

### ۱. تفاوت اصلی با NHA

| جنبه | NHA | HALO-ReGLA |
|---|---|---|
| Dense branch | Local window | **Global dense** |
| Linear branch | RFF | **ELU+1 kernel** |
| Gate bias init | 3.0 | 1.8 |
| Cumsum | ندارد | **دارد** |

### ۲. Dense + Linear به صورت موازی

HALO-ReGLA **دو استریم** را همزمان محاسبه می‌کند:

$$
O_i = g_i \cdot O^\text{dense}_i + (1 - g_i) \cdot O^\text{lin}_i
$$

### ۳. Gate یادگیرنده per-head

هر head یک گیت مستقل دارد که می‌تواند به سمت dense یا linear برود.

**مقدار اولیه:** $\sigma(1.8) \approx 0.86$ → dense غالب.

### ۴. Kernel انتخاب شده

$$
\phi(x) = \mathrm{elu}(x) + 1
$$

**مزایا:**
- Non-negative → denominator مثبت
- Smooth → gradient خوب
- Bounded → بدون overflow

### ۵. Causality با Prefix Sum

در linear stream:

$$
\text{cum}^{(i)} = \sum_{j \le i} \phi(K_j) V_j^\top
$$

این به صورت خودکار causality را حفظ می‌کند.

### ۶. بدون Speedup واقعی

**مهم:** HALO-ReGLA هر دو stream را محاسبه می‌کند. برای speedup واقعی:

1. Train با هر دو
2. شناسایی heads که $\bar{g}_h < 0.1$
3. تبدیل به pure linear

این limitation را **صادقانه** ذکر کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_halo_regla/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**