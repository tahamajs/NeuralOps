# Pre-Submission Checklist — FASA

## Setup
- [ ] Copied `_TEMPLATE/` to `fasa/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module fasa`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] States FASA is token pruning, not attention replacement
- [ ] States decoder path skips pruning
- [ ] Reports all three model results
- [ ] Honest about variance being a weak signal

### Introduction
- [ ] Motivates token pruning
- [ ] Lists two constraints (test-time, no-causal)
- [ ] Contributions honest

### Method
- [ ] Variance formula clear
- [ ] CLS pinning explained
- [ ] Decoder path documented
- [ ] Complexity analysis distinguishes FLOPs from wall-clock

### Results
- [ ] All tables filled
- [ ] Keep ratio ablation
- [ ] Importance method ablation
- [ ] CLS protection ablation

### Analysis
- [ ] Which tokens pruned
- [ ] Correlation between variance and attention
- [ ] CLS protection analysis
- [ ] Decoder no-op documented
- [ ] Overhead breakdown

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Mathematical derivations
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (workshop or TMLR)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE fasa
```

### مرحله ۲: پر کردن `variables.json`

اگر FASA را اجرا کردی، نتایج را از `results/unified_fasa/*/summary.csv` بردار. اگر نه، از اعداد پیش‌فرض (که بر اساس انتظار تخمین زده شده).

### مرحله ۳: Apply و build

```powershell
cd papers\fasa
python apply_vars.py --module fasa
python build.py --module fasa
```

---

## ⚠️ نکات مخصوص FASA

### ۱. چهار کشف مهم

**کشف ۱ — Token pruning است، نه attention mechanism:**
FASA توکن‌های K/V رو بر اساس واریانس prune می‌کند. این یک روش **post-hoc** است.

**کشف ۲ — Decoder path هیچ pruning نمی‌کند:**
```python
if is_causal or is_cross:
    # Standard full attention — NO pruning!
```
برای GPT-2، فقط LoRA اضافه می‌کند. **Delta = 0.00%**.

**کشف ۳ — CLS pinning حیاتی است:**
```python
importance[:, 0] = 1e9
```
بدون این، مدل collapse می‌کند (چون CLS توکن برای classification حیاتی است).

**کشف ۴ — "MoSKA" در نام توضیح داده نشده:**
`mixattention_moska` به عنوان alias ثبت شده ولی معنا ندارد. این کلاس واقعاً **FASA** است.

### ۲. اگر بعداً pruning رو بهتر کردی

- Learned importance head
- Layer-wise keep ratios
- Causal prefix scan

بخش‌های `08_analysis.md` و `10_limitations.md` رو به‌روز کن.

### ۳. Venue مناسب

- **TMLR** (honest ML research)
- **Workshops** (efficiency)
- **arXiv** همیشه

### ۴. تفاوت با Cosformer و GoldFinch

| جنبه | FASA | Cosformer | GoldFinch |
|---|---|---|---|
| نوع | Token pruning | Linear attention | Hybrid |
| Decoder | **No prune** | Real linear | Sliding window |
| Encoder | Prune | Linear state | Sliding + fake long |
| Accuracy (D) | احتمالاً -6% | -3% | -3.7% |
| پارامتر | LoRA only | LoRA only | LoRA + gate + linear |

**FASA ماژول "متوسط" است** — نه collapse (مثل Galerkin)، نه صفر (مثل GatedDeltaNet)، نه بسیار خوب (مثل Cosformer). یک baseline قابل قبول برای token pruning.

### ۵. مسئله اصلی برای مقاله

**باید شفاف بگی که decoder path کار نمی‌کند.** این نه یک باگ است نه یک ویژگی — یک **محدودیت طراحی** است. برای causal decoderها FASA عملاً یک adapter با LoRA است، نه یک روش sparse attention.

---

**قدم بعدی: اگر نتایج FASA را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**