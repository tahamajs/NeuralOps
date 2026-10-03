# Pre-Submission Checklist — Compressed MHA

## Setup
- [ ] Copied `_TEMPLATE/` to `compressed_mha/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module compressed_mha`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Joint SVD mentioned
- [ ] Parameter reduction stated
- [ ] Nested detection fix mentioned
- [ ] Multi-architecture support

## Method
- [ ] Joint SVD formula
- [ ] Parameter count formula
- [ ] Nested detection rule
- [ ] LoRA on both branches
- [ ] Causality explanation

## Results
- [ ] Main results (Table 7.1)
- [ ] Parameter counts (Table 7.2)
- [ ] Per-layer breakdown
- [ ] Rank ablation
- [ ] Joint vs independent
- [ ] Nested detection validation

## Honesty
- [ ] Parameter reduction is real
- [ ] Accuracy delta honest
- [ ] Nested bug documented
- [ ] Single-seed limitation

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/compressed_mha
nano papers/compressed_mha/variables.json
cd papers/compressed_mha
python apply_vars.py --module compressed_mha
grep -rn "{{" .
python build.py --module compressed_mha --pdf
```
```

---

## 🎯 خلاصه نهایی

### ویژگی‌های منحصر به فرد Compressed MHA

1. **Joint QKV SVD** — اشتراک basis بین Q، K، V
2. **Separate O SVD** — فشرده‌سازی جداگانه W_O
3. **Real parameter reduction** — `4d²` → `2r(d+d_h)`
4. **Nested detection fix** — چک `out_lin` **قبل از** nested
5. **Safe failure** — `logger.warning` به جای `raise`
6. **Frozen SVD + LoRA** — ترکیب فشرده‌سازی و recovery

### تفاوت با مقالات قبلی

| مقاله | فشرده‌سازی | پارامتر |
|---|---|---|
| NHA | attention جایگزین | + |
| LEKAv3 | landmark | + |
| Linformer | sequence projection | + |
| A3/AASVD | weight SVD | - |
| **Compressed MHA** | **joint weight SVD** | **-** |

### نکات کلیدی در مقاله

1. **Joint SVD proof** — چرا shared basis بهتر است
2. **Nested detection bug** — چطور 50.92% اتفاق می‌افتاد
3. **Parameter accounting** — صادقانه
4. **Multi-architecture** — ۴ مدل

### هشدار مهم در `variables.json`

- `RANK_QKV = 64` (پیش‌فرض)
- `RANK_OUT = 64`
- **Default** در کد 384 است که **هیچ compression نمی‌دهد**
- برای کاهش واقعی، `--rank_qkv 64` یا کمتر بده

---

**قدم بعدی: `run_all_4_compressed_mha.py` را با `--rank_qkv 64 --rank_out 64` اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**