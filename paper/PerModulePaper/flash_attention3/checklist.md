# Pre-Submission Checklist — FlashAttention-3

## Setup
- [ ] Copied `_TEMPLATE/` to `flash_attention3/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module flash_attention3`
- [ ] No `{{PLACEHOLDER}}` remains

## Abstract
- [ ] Under 250 words
- [ ] **Exact attention claim stated**
- [ ] **No compression claim stated**
- [ ] Hardware dependency mentioned
- [ ] Fallback path mentioned
- [ ] FP8 optional path mentioned

## Intro
- [ ] Motivation for IO-aware attention
- [ ] FA3 distinctives (TMA, warp spec)
- [ ] Contributions honest

## Method
- [ ] Tiled algorithm described
- [ ] Dispatch logic shown
- [ ] Fallback explained
- [ ] FP8 mode
- [ ] **Parameter analysis honest**
- [ ] Nested detection note

## Results
- [ ] Main accuracy table
- [ ] **Execution path table**
- [ ] Sequence scaling
- [ ] Output correctness
- [ ] FP8 ablation

## Honesty
- [ ] **"No compression" stated in abstract, intro, discussion, limitations**
- [ ] **Hardware dependency stated**
- [ ] **Fallback is SDPA (functionally identical)**
- [ ] FP8 not tested broadly
- [ ] Single-seed
- [ ] Not tested on Hopper directly

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] FA3 installation instructions
- [ ] Path used documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/flash_attention3
nano papers/flash_attention3/variables.json
cd papers/flash_attention3
python apply_vars.py --module flash_attention3
grep -rn "{{" .
python build.py --module flash_attention3 --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **صداقت درباره «compression»**

این ماژول **compression نیست**. پارامتر کاهش نمی‌ده. فقط:
- سرعت (روی Hopper)
- حافظه (روی همه GPUها)

### ۲. **Fallback به SDPA**

روی RTX 4070 (Ampere) که FA3 نداره، به SDPA fallback می‌کنه. این یعنی:
- accuracy = baseline (exact)
- speed ≈ baseline (SDPA هم بهینه است)
- **تفاوتی در نتایج بنچمارک دیده نمی‌شه**

### ۳. **باید روی H100 تست بشه**

نتایج واقعی روی Hopper جالبه. روی Ampere، فقط baseline.

### ۴. **FP8 ریسک داره**

فقط روی Hopper کار می‌کنه، و accuracy رو کم می‌کنه.

### ۵. **مزیت اصلی: Exact + Fast**

- Linear attention: approximate + fast
- MoE: approximate + slow
- **FA3: exact + fast** (روی Hopper)

---

**قدم بعدی: اگر دسترسی به H100 داری، روی اون تست کن. اگر نه، نتایج Ampere رو با `Path used: SDPA` صادقانه گزارش کن.**