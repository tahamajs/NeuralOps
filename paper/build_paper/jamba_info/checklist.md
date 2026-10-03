# Pre-Submission Checklist — JambaInfo

## Setup
- [ ] Copied `_TEMPLATE/` to `jamba_info/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module jamba_info`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Information bottleneck motivation clear
- [ ] Bottleneck dimension stated
- [ ] Parameter increase stated
- [ ] Not a compression method

## Intro
- [ ] IB motivation
- [ ] Concept bottleneck explained
- [ ] Differential attention explained
- [ ] Contributions listed

## Method
- [ ] IB formula
- [ ] Bottleneck formula
- [ ] Differential formulas
- [ ] Fusion formula
- [ ] Parameter analysis

## Results
- [ ] Main accuracy
- [ ] Parameter table (honest)
- [ ] Bottleneck dim ablation
- [ ] No-bottleneck ablation
- [ ] α init ablation
- [ ] Differential ablation
- [ ] LoRA + recovery ablations

## Honesty
- [ ] Parameter increase stated
- [ ] Sparsity is placeholder
- [ ] Soft fallback noted
- [ ] Single-seed stated
- [ ] Cross-attention not supported

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Bottleneck dim documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/jamba_info
nano papers/jamba_info/variables.json
cd papers/jamba_info
python apply_vars.py --module jamba_info
grep -rn "{{" .
python build.py --module jamba_info --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **مزیت اصلی: Information Bottleneck**

این ماژول **اولین** architecture hybrid هست که از IB theory برای SSM branch استفاده می‌کنه. این را در:
- Abstract
- Intro
- Method (با جزئیات)
- Discussion

برجسته کن.

### ۲. **صداقت درباره پارامتر**

پارامتر **۲۸٪ بیشتر** می‌شه. این را واضح بگو.

### ۳. **دو نوآوری واقعی**

1. **Concept bottleneck** برای SSM (با GELU)
2. **IB-motivated design** (تئوری پشتش)

### ۴. **α init = σ(-2)**

برخلاف HymbaTransX (σ(-1))، این ماژول با **attention bias** شروع می‌کنه. دلیلش: bottleneck باعث می‌شه SSM اول ضعیف‌تر باشه.

### ۵. **Fusion order متفاوت**

`(1-α) A + α M` — توجه کن که برعکس HymbaTransX هست.

---

**قدم بعدی: `run_all_4_jamba_info.py` را با `--bottleneck_dim 256` اجرا کن، اعداد را از `summary.csv` بگیر، در `variables.json` بگذار.**