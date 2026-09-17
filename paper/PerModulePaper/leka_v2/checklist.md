# Pre-Submission Checklist — LEKAv2

## Setup
- [ ] Copied `_TEMPLATE/` to `leka_v2/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module leka_v2`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Landmark distillation stated
- [ ] Dual-scale RFF stated
- [ ] Softpick mentioned
- [ ] Parameter increase stated
- [ ] Not truly linear at ρ=0.5 stated

## Intro
- [ ] Linear attention limitations
- [ ] Landmark distillation motivation
- [ ] Contributions

## Method
- [ ] Dual-scale feature formula
- [ ] Landmark selection algorithm
- [ ] Softpick formula
- [ ] Local-global fusion
- [ ] Complexity analysis (honest about quad)
- [ ] Parameter analysis

## Results
- [ ] Main accuracy
- [ ] Parameter table
- [ ] Speed vs L
- [ ] ρ ablation
- [ ] F ablation
- [ ] Landmark ablation
- [ ] Window ablation
- [ ] Softpick ablation
- [ ] Dual-scale ablation
- [ ] LoRA ablation
- [ ] Recovery ablation
- [ ] Gate statistics

## Honesty
- [ ] Not truly linear at ρ=0.5 stated
- [ ] Fatal error handling noted
- [ ] Single-seed stated
- [ ] Cross-attention not supported

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] All hyperparameters documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/leka_v2
nano papers/leka_v2/variables.json
cd papers/leka_v2
python apply_vars.py --module leka_v2
grep -rn "{{" .
python build.py --module leka_v2 --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **ماهیت: Landmark + Local hybrid**

این ماژول **نه خطی خالص** است و **نه attention معمولی**. ترکیبی از:

- **Local sliding window** — O(L·W)
- **Landmark global attention** — O(L·L_max) که با ρ=0.5 هنوز **quadratic** است

**باید صادق باشی** که این ماژول در ρ=0.5 خطی نیست.

### ۲. **چهار نوآوری واقعی**

1. **Landmark distillation** با scorer یادگیرنده
2. **Dual-scale RFF** (F + F/2)
3. **Softpick** — softmax مقاوم به mask
4. **Per-head correction matrix** و **local gate**

### ۳. **صداقت درباره پیچیدگی**

باید در abstract و limitations بگی که این ماژول با ρ=0.5 هنوز **quadratic** است.

### ۴. **باگ‌های قابل ذکر در limitations**

1. `raise RuntimeError` در `copy_weights`
2. landmark attention با L_max=ρL هنوز quadratic
3. Corrections matrix scaling $O(H d_h^2)$

### ۵. **تست سریع**

```bash
python run_all_4_leka_v2.py --models distilbert,roberta,gpt2 --feature_dim 128 --landmark_ratio 0.5 --batch_size 4
```

---

**قدم بعدی: `run_all_4_leka_v2.py` را اجرا کن، اعداد را از `summary.csv` بگیر، در `variables.json` بگذار.**