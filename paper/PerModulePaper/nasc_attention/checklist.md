# Pre-Submission Checklist — NASC

## Setup
- [ ] Copied `_TEMPLATE/` to `nasc/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nasc`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Three-branch cascade described
- [ ] Batch-level gate disclosed
- [ ] `softpick` fallback disclosed
- [ ] `inject_nasc` raise disclosed

### Introduction
- [ ] Motivates hybrid attention
- [ ] Discloses three caveats
- [ ] Contributions honest

### Method
- [ ] All three branches detailed
- [ ] Gate formula clear
- [ ] Complexity analysis
- [ ] Memory analysis for RFF

### Results
- [ ] All tables filled
- [ ] Gate distribution shown
- [ ] Branch ablation
- [ ] Window/Block/TopK/RFF ablations

### Analysis
- [ ] Encoder default analyzed
- [ ] Gate interpretation
- [ ] Branch ablation
- [ ] softpick fallback tested
- [ ] RFF memory computed
- [ ] Sparse loop overhead

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Math derivations
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
Copy-Item -Recurse _TEMPLATE nasc
```

### مرحله ۲: پر کردن `variables.json`

### مرحله ۳: Apply و build

```powershell
cd papers\nasc
python apply_vars.py --module nasc
python build.py --module nasc
```

---

## ⚠️ نکات مخصوص NASC

### ۱. شش کشف مهم

**کشف ۱ — سه شاخه:** local + sparse block + spectral RFF

**کشف ۲ — گیت per-batch است:** `hidden_states.mean(dim=1)` یعنی یک بردار `g` برای تمام توکن‌های batch.

**کشف ۳ — `softpick` حالا fix شده:** نسبت به AEGIS V4/V5، این نسخه fallback به softmax دارد. ولی **batch-level** است.

**کشف ۴ — Encoder defaults to dense:** مگر اینکه `enable_cascade_on_encoders=True` بگذاری.

**کشف ۵ — RFF causal path memory:** `KV_cumsum` خیلی بزرگ می‌شود. برای L=128 حدود 50 MB. برای L=1024 حدود 3.2 GB → OOM.

**کشف ۶ — `inject_nasc` دوباره raise می‌کند:** با سایر ماژول‌ها (که `return None` می‌کنند) متناقض است. این **robustness regression** است.

### ۲. اگر بعداً fix کردی

- `inject_nasc` را `return None` کن
- Per-token gating اضافه کن
- Sparse block را vectorize کن
- Fused RFF kernel بنویس

### ۳. Venue مناسب

- **TMLR** (honest ML research)
- **Workshops** (efficiency)
- **arXiv** همیشه

### ۴. تفاوت با سایر ماژول‌ها

| ماژول | پیچیدگی | Path |
|---|---|---|
| FASA | Token pruning | Encoder-only prune |
| Cosformer | Linear attention | Both |
| NASC | 3-branch hybrid | Decoder cascade |
| GoldFinch | Hybrid + preservation | Both |

**NASC پیچیده‌ترین ماژول توست** — سه شاخه، گیت، RFF. ولی این پیچیدگی هم overhead دارد.

### ۵. مسئله اصلی برای مقاله

- **Encoder path عملاً dense است.** NASC روی DistilBERT/RoBERTa هیچ کاری نمی‌کند (مگر اینکه cascade صریحاً فعال شود).
- **GPT-2 cascade فعال است** ولی گیت به local متمایل است (`g0 ≈ 0.72`) — یعنی sparse و global به‌سختی استفاده می‌شوند.

---

**قدم بعدی: اگر نتایج NASC را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**