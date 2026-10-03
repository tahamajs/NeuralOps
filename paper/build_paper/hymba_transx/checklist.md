# Pre-Submission Checklist — HymbaTransX

## Setup
- [ ] Copied `_TEMPLATE/` to `hymba_transx/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module hymba_transx`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Differential attention stated
- [ ] SSM fusion stated
- [ ] **Parameter increase stated explicitly**
- [ ] Not a compression method

## Intro
- [ ] Motivation for differential attention
- [ ] Motivation for SSM fusion
- [ ] Contributions listed

## Method
- [ ] Two Q-K pairs formula
- [ ] Differential ReLU formula
- [ ] SSM recurrence
- [ ] α fusion
- [ ] Parameter analysis honest

## Results
- [ ] Main accuracy table
- [ ] Parameter table (honest, increase)
- [ ] Differential ablation
- [ ] SSM fusion ablation
- [ ] α values table
- [ ] LoRA ablation
- [ ] Recovery ablation

## Honesty
- [ ] Parameter increase stated in abstract
- [ ] Not a compression method stated
- [ ] Soft fallback risk noted
- [ ] Single-seed stated
- [ ] Cross-attention not supported

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] λ and α init documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/hymba_transx
nano papers/hymba_transx/variables.json
cd papers/hymba_transx
python apply_vars.py --module hymba_transx
grep -rn "{{" .
python build.py --module hymba_transx --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **ماهیت: Hybrid نه Compression**

HymbaTransX پارامتر **کم نمی‌کنه** — بلکه **اضافه** می‌کنه. باید در:
- Abstract
- Intro
- Limitations
- Discussion

این را واضح بگویی.

### ۲. **دو نوآوری واقعی**

1. **Differential attention** با λ=0.2 (نه 0.8) برای hybrid
2. **Learned α fusion** با init σ(-1)≈0.27

### ۳. **تفاوت با Hymba اصلی**

Hymba از attention معمولی استفاده می‌کنه، ما از **differential**. این را برجسته کن.

### ۴. **سه رفتار α**

- DistilBERT: α → کوچک (SSM قوی‌تر)
- RoBERTa: α → بزرگ (attention قوی‌تر)
- GPT-2: α ≈ 0.5 (متوسط)

این رو در Analysis تحلیل کن.

### ۵. **HeadRoPE fallback**

یک نوآوری جانبی — وقتی مدل parent RoPE نمی‌ده، از HeadRoPE داخلی استفاده می‌کنه.

---

**قدم بعدی: `run_all_4_hymba_transx.py` را اجرا کن، اعداد را از `summary.csv` بگیر (شامل α نهایی)، در `variables.json` بگذار.**