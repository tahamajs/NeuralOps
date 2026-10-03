# Pre-Submission Checklist — DecayLinearAttention

## Setup
- [ ] Copied `_TEMPLATE/` to `decay_linear_attention/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module decay_linear_attention`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **O(1) memory claim stated**
- [ ] **Data-dependent decay mentioned**
- [ ] Cross-attention support mentioned
- [ ] Chunked scan described

## Intro
- [ ] Motivation for data-dependent decay
- [ ] Two problems identified (decay + memory)
- [ ] Contributions listed

## Method
- [ ] Decay formula
- [ ] Chunked scan algorithm
- [ ] ELU+1 feature map
- [ ] **Memory complexity explicit**
- [ ] Cross-attention formula

## Results
- [ ] Main accuracy table
- [ ] **State memory table**
- [ ] Inference time table
- [ ] Chunk size ablation
- [ ] Log-decay range ablation
- [ ] Feature map ablation
- [ ] Learned decay table

## Honesty
- [ ] **State memory advantage stated clearly**
- [ ] Cross-attention not fully tested (stated)
- [ ] Single-seed limitation
- [ ] No theoretical bound stated
- [ ] Feature map simplicity acknowledged

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Chunk size documented
- [ ] Log-decay range documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/decay_linear_attention
nano papers/decay_linear_attention/variables.json
cd papers/decay_linear_attention
python apply_vars.py --module decay_linear_attention
grep -rn "{{" .
python build.py --module decay_linear_attention --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **مزیت اصلی: O(1) State Memory**

این ماژول **حافظه‌ی حالت O(1)** داره (16 KB در مقایسه با CHELA-SLAY که 8 GB داشت). این یک مزیت **عظیم** است — تقریباً **500,000× کمتر**!

### ۲. **Data-Dependent Decay**

هر head یک decay rate یاد می‌گیره که horizon متفاوتی می‌ده. این الهام‌گرفته از:
- HGRN (forget gate)
- Mamba (selective SSM)
- RetNet (fixed decay)

### ۳. **Chunked Scan**

بجای cumsum روی کل L، در هر chunk یک state نگه می‌داره:
- Intra-chunk: dense causal attention در chunk
- Inter-chunk: state-passing

### ۴. **Cross-Attention**

این ماژول **cross-attention** رو پشتیبانی می‌کنه (بسیاری از ماژول‌های دیگه ندارن). فقط forward scan استفاده می‌شه.

### ۵. **صداقت درباره محدودیت‌ها**

- Feature map ساده (ELU+1)
- Cross-attention فقط preliminary
- Single-seed

### ۶. **تست سریع**

```bash
python run_all_4_decay_linear.py --models distilbert,roberta,gpt2 --chunk_size 32 --batch_size 8
```

برخلاف CHELA-SLAY، این ماژول **OOM نمی‌ده** چون state memory کوچیکه.

---

**قدم بعدی: `run_all_4_decay_linear.py` را اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**