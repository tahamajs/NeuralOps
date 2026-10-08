# Pre-Submission Checklist — TokenCompressedAttention

## Setup
- [ ] Copied `_TEMPLATE/` to `token_compressed/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module token_compressed`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] KV cache reduction stated
- [ ] **Parameter increase stated explicitly**
- [ ] Learned scoring mentioned

## Method
- [ ] Score function formula present
- [ ] Causal mask derivation
- [ ] [CLS] preservation explained
- [ ] Parameter analysis honest

## Results
- [ ] KV length table
- [ ] ρ ablation
- [ ] Score net ablation
- [ ] [CLS] ablation
- [ ] Comparison with H2O/StreamingLLM

## Honesty
- [ ] Parameter increase is stated in abstract
- [ ] Parameter increase is stated in results
- [ ] "Compression" refers only to KV cache, not weights
- [ ] Failure modes documented
- [ ] Single-seed limitation stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Score net params documented

## Final
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/token_compressed
nano papers/token_compressed/variables.json
cd papers/token_compressed
python apply_vars.py --module token_compressed
grep -rn "{{" .
python build.py --module token_compressed --pdf
```
```

---

## 🎯 خلاصه

### تفاوت‌های کلیدی این مقاله با A3/AASVD

1. **نوع فشرده‌سازی:** token-level، نه weight-level
2. **پارامتر:** **افزایش** می‌یابد نه کاهش (باید صادق باشی)
3. **KV cache:** کاهش واقعی ۵۰٪
4. **Compute:** کاهش واقعی ۵۰٪
5. **Baselines:** H2O, StreamingLLM, SnapKV (نه A3)

### نکته مهم در `variables.json`

مقدار `RATIO` را روی **۰.۵** گذاشتم چون:
- ρ = ۰.۵ یک sweet spot است
- کاهش KV cache = ۵۰٪
- افت دقت ~۲-۳٪

اگر می‌خواهی محافظه‌کارتر باشی، `ρ=0.7` بگذار.
اگر تهاجمی‌تر، `ρ=0.3`.

### برجسته کردن در مقاله

- **`[CLS]` preservation** — یک نوآوری ساده ولی مهم
- **Causal mask against original positions** — یک جزئیات فنی مهم
- **Learnable score** vs fixed heuristics — مزیت اصلی
- **Honest parameter count** — باید واضح بگویی که افزایش دارد

---

**قدم بعدی: `run_all_4_token_compressed.py` را با `compression_ratio=0.5` اجرا کن، اعداد را از `summary.csv` بردار، و در `variables.json` بگذار.**