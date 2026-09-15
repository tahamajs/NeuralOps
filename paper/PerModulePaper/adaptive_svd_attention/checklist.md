# Pre-Submission Checklist — Adaptive SVD Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_svd/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_svd`
- [ ] No `{{PLACEHOLDER}}` remains

## Abstract
- [ ] Under 250 words
- [ ] Energy-thresholded rank stated
- [ ] Zero-shot claim stated
- [ ] All-four-projection claim stated

## Intro
- [ ] Motivation for dynamic rank
- [ ] Problem statement formal
- [ ] Contributions listed

## Method
- [ ] Energy formula
- [ ] Symmetric split formula
- [ ] Residual adapter formula
- [ ] Rank selection algorithm
- [ ] Parameter analysis

## Results
- [ ] Main accuracy table
- [ ] Per-projection rank table
- [ ] Energy τ ablation
- [ ] Residual rank ablation
- [ ] Fixed vs adaptive ablation

## Honesty
- [ ] Attention-only compression stated
- [ ] Total reduction stated
- [ ] Single-seed stated
- [ ] T5 not tested (if applicable)

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] No calibration data required

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/adaptive_svd
nano papers/adaptive_svd/variables.json
cd papers/adaptive_svd
python apply_vars.py --module adaptive_svd
grep -rn "{{" .
python build.py --module adaptive_svd --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **مزیت اصلی: Energy Thresholding**

برخلاف SOA Fused Attention که rank ثابت داره، این ماژول **rank پویا** انتخاب می‌کنه بر اساس energy threshold. این را در abstract و intro برجسته کن.

### ۲. **All Four Projections**

برخلاف SOA Fused که فقط Q/K/V رو compress می‌کنه، این W_O رو هم compress می‌کنه. این یعنی **reduction بیشتر**.

### ۳. **T5 و Cross-Attention**

این ماژول از T5 relative bias و cross-attention پشتیبانی می‌کنه — چیزی که SOA Fused نداره. این را در discussion برجسته کن.

### ۴. **Symmetric Split**

استفاده از √Σ در هر دو factor (نه asymmetric) — توضیح بده چرا.

### ۵. **تست سریع**

```bash
python run_all_4_adaptive_svd.py --models distilbert,roberta,gpt2 --energy_threshold 0.92 --residual_rank 16
```

---

**قدم بعدی: `run_all_4_adaptive_svd.py` را اجرا کن، اعداد را از `summary.csv` و `layer_stats` بگیر، در `variables.json` بگذار. اگر خواستی من می‌توانم `variables.json` را با اعداد واقعی از لاگ‌هایت پر کنم.**
