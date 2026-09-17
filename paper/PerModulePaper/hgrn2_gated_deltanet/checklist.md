# Pre-Submission Checklist — HGRN2-GatedDeltaNet

## Setup
- [ ] Copied `_TEMPLATE/` to `hgrn2_gated_deltanet/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module hgrn2_gated_deltanet`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Linear complexity stated
- [ ] **Parameter increase stated explicitly**
- [ ] Short-sequence slowdown stated
- [ ] Breakeven L mentioned

## Intro
- [ ] Two-branch motivation clear
- [ ] Contributions listed
- [ ] Organization complete

## Method
- [ ] HGRN-2 formulas
- [ ] DeltaNet formulas
- [ ] Gate formula
- [ ] Parameter analysis honest
- [ ] Failure handling documented

## Results
- [ ] Main accuracy table
- [ ] Parameter table (honest)
- [ ] Speed vs L table
- [ ] Gate ablation
- [ ] Branch ablation
- [ ] Recovery steps ablation

## Honesty
- [ ] Parameter increase stated in abstract, results, discussion
- [ ] Short-sequence slowdown stated
- [ ] Python loop limitation stated
- [ ] RoBERTa instability documented
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Batch_size=4 note for RoBERTa

## Final
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/hgrn2_gated_deltanet
nano papers/hgrn2_gated_deltanet/variables.json
cd papers/hgrn2_gated_deltanet
python apply_vars.py --module hgrn2_gated_deltanet
grep -rn "{{" .
python build.py --module hgrn2_gated_deltanet --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **صداقت درباره پارامتر**

این ماژول **پارامتر کم نمی‌کند** — بلکه **زیاد** می‌کند (به‌خاطر دو branch و gate). باید این را در abstract، results، و limitations برجسته کنی.

### ۲. **صداقت درباره سرعت**

در توالی‌های کوتاه (L=128 که SST-2 است)، این ماژول **کندتر** است. فقط در L ≥ 512 سریع‌تر می‌شود. این را هم باید صادقانه بگویی.

### ۳. **سه نوآوری واقعی**

- **دو branch مکمل** (element-wise + matrix)
- **Gate یادگیرنده** (per-token)
- **Linear complexity** واقعی

### ۴. **OOM روی RoBERTa**

پیشنهاد: `--batch_size 4` و `--lora_rank 8`.

### ۵. **محاسبه breakeven**

باید در `variables.json` مقداری بگذاری که از لاگ واقعی بیاید. اگر L* = 512 است، این یعنی SST-2 (با L=128) در ناحیه کند است.

---

**قدم بعدی: `run_all_4_hgrn2_gated_deltanet.py` را با تنظیمات کم‌حافظه اجرا کن، اعداد را بگیر، در `variables.json` بگذار. اگر خواستی، من می‌توانم `variables.json` را با اعداد واقعی از لاگ‌هایت پر کنم — فقط CSV را بفرست.**