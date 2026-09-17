# Pre-Submission Checklist — GatedDeltaNetBlock

## Setup
- [ ] Copied `_TEMPLATE/` to `gated_deltanet_block/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module gated_deltanet_block`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Honest Reporting
- [ ] Abstract states LayerNorm is fresh (not pretrained)
- [ ] Abstract states silent failure path
- [ ] Abstract reports non-causal MHA for GPT-2
- [ ] Section 8.1 fully analyzes LayerNorm mismatch
- [ ] Section 8.2 covers silent try/except
- [ ] Section 8.4.1 documents non-causal MHA
- [ ] Section 10 lists all limitations honestly

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Both paths mentioned
- [ ] Honest about negative results

### Introduction
- [ ] Motivates block-level vs attention-level
- [ ] Contributions include negative result

### Method
- [ ] Block structure clear
- [ ] LayerNorm problem documented
- [ ] Weight transfer limitations detailed

### Results
- [ ] Both paths reported
- [ ] LayerNorm statistics analyzed
- [ ] Silent failure detection documented

### Analysis
- [ ] LayerNorm error decomposition
- [ ] Silent failure analysis
- [ ] Non-causal MHA analysis
- [ ] Nested encoder inconsistency

## Reproducibility
- [ ] Code on GitHub
- [ ] Logging added for weight transfer
- [ ] Seeds documented

## Appendices
- [ ] A: Mathematical derivation
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Complete negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (TMLR for negative results)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE gated_deltanet_block
```

### مرحله ۲: بررسی نتایج قبلی

از لاگ‌ها:
- **Path A (MHA):** اگر OOM نداده، حدود +۰.۸۸٪ پارامتر، دقت مشابه baseline
- **Path B (DeltaNet):** timeout (~۹۰ ثانیه per 5 steps)، OOM روی RoBERTa/GPT-2

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و build

```powershell
cd papers\gated_deltanet_block
python apply_vars.py --module gated_deltanet_block
python build.py --module gated_deltanet_block
```

---

## ⚠️ نکات مخصوص GatedDeltaNetBlock

### ۱. سه باگ اصلی که باید برجسته کنی

1. **LayerNorm pretrained نیست** → mismatch در توزیع activation
2. **`try/except: pass`** → شکست‌های silent
3. **`nn.MultiheadAttention` بدون causal mask** → خرابی GPT-2

### ۲. تفاوت با gated_delta_net (attention-level)

| جنبه | attention-level | block-level |
|---|---|---|
| LayerNorm | pretrained حفظ می‌شود | **new** |
| Residual | unchanged | **new** |
| Nested handling | automatic | manual |
| Silent failures | fewer | **more** |

### ۳. توصیه اصلی

**Block-level replacement را توصیه نکن مگر اینکه:**
- LayerNorm pretrained داشته باشی
- روی مدل‌های flat (نه nested) کار کنی
- causal masking را صریحاً اضافه کنی
- `try/except` را با logging جایگزین کنی

### ۴. اگر بعداً fix کردی

- `D_LN_FRESH` رو با عدد بهتر عوض کن (با pretrained LN)
- `G_MHA_ACC` رو با عدد واقعی عوض کن (با causal mask)
- `D_MHA_ACC` رو با عدد جدید
- بخش‌های critical در `08_analysis.md` رو با اعداد جدید به‌روز کن

### ۵. Venue

- **TMLR** برای negative results
- **arXiv** همیشه
- **Workshops** (efficiency/compression workshops)

---

**قدم بعدی: اگر خواستی، `variables.json` رو با اعداد دقیق از لاگ‌های خودت پر می‌کنم. فقط لاگ‌های `gated_deltanet_block` رو بفرست.**