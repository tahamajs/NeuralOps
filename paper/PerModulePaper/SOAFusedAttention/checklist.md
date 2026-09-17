# Pre-Submission Checklist — SOA Fused Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `soa_fused/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module soa_fused`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Zero-shot claim stated
- [ ] Real parameter reduction stated
- [ ] Rank and ratio mentioned

## Intro
- [ ] Motivation clear (attention dominates params)
- [ ] Zero-shot advantage highlighted
- [ ] Contributions listed

## Method
- [ ] SVD factorization formula
- [ ] Asymmetric split explained
- [ ] Rank selection formula
- [ ] Parameter analysis honest

## Results
- [ ] Main accuracy table
- [ ] Attention parameter table
- [ ] Per-projection breakdown
- [ ] Rank ablation
- [ ] LoRA rank ablation
- [ ] Recovery ablation

## Honesty
- [ ] Attention-only compression stated
- [ ] $W_O$ kept intact explained
- [ ] Total reduction stated
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] No calibration data required (highlight!)

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/soa_fused
nano papers/soa_fused/variables.json
cd papers/soa_fused
python apply_vars.py --module soa_fused
grep -rn "{{" .
python build.py --module soa_fused --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **مزیت اصلی: Zero-Shot**

این ماژول **نیازی به calibration data نداره** — یک مزیت بزرگ نسبت به A3 و AASVD. این را در:
- Abstract
- Intro
- Discussion
- Comparison tables

برجسته کن.

### ۲. **صداقت درباره دامنه**

فقط Q/K/V فشرده می‌شن، نه W_O و نه FFN. پس:
- **Attention reduction: 3×**
- **Total model reduction: ~1.02×**
- این را واضح بگو

### ۳. **$W_O$ چرا حفظ می‌شود**

در Method و Analysis توضیح بده که چرا فشرده‌سازی W_O خطای compound ایجاد می‌کنه.

### ۴. **Composition با روش‌های دیگر**

SOA Fused Attention با GQA، TCA، KV quantization ترکیب می‌شه. این را در Discussion برجسته کن.

### ۵. **تست سریع**

```bash
python run_all_4_soa_fused.py --models distilbert,roberta,gpt2 --rank 128
```

اگر OOM داد: `--batch_size 4`.

---

**قدم بعدی: `run_all_4_soa_fused.py` را اجرا کن، اعداد واقعی را از `summary.csv` بگیر، در `variables.json` بگذار.**