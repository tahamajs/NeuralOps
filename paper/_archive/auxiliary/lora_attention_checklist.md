# Pre-Submission Checklist — LoRA Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `lora_attention/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module lora_attention`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicit:** this does NOT reduce parameters
- [ ] Trainable reduction stated
- [ ] Multi-architecture mentioned
- [ ] Mergeability mentioned

## Method
- [ ] LoRA formula
- [ ] Weight transfer for all 4 architectures
- [ ] Nested detection rule
- [ ] Merge procedure
- [ ] Zero-init proof

## Results
- [ ] Main results (Table 7.1)
- [ ] Trainable params (Table 7.2)
- [ ] Which projections ablation
- [ ] Rank ablation
- [ ] Merge verification
- [ ] Compression recovery

## Honesty
- [ ] No compression claim
- [ ] Parameter increase stated
- [ ] Mergeability verified
- [ ] Single-seed limitation
- [ ] Hard-fail behavior documented

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/lora_attention
nano papers/lora_attention/variables.json
cd papers/lora_attention
python apply_vars.py --module lora_attention
grep -rn "{{" .
python build.py --module lora_attention --pdf
```
```

---

## 🎯 خلاصه نهایی

### ویژگی‌های منحصر به فرد LoRA Attention

1. **Multi-architecture** — ۴ مدل
2. **Zero inference overhead** — mergeable
3. **Attention-only** — نه FFN
4. **Compression-compatible** — recovery mechanism
5. **Safe nested detection** — چک `self.query` اول

### تفاوت با مقالات قبلی

| مقاله | نوع | پارامتر |
|---|---|---|
| A3/AASVD | compression | کاهش |
| Compressed MHA | compression | کاهش |
| TCA | token compression | افزایش |
| **LoRA Attention** | **PEFT** | **افزایش (قبل از merge)** |

### برجسته کردن در مقاله

1. **این compression نیست** — PEFT است
2. **Mergeability** — مزیت کلیدی
3. **Attention vs FFN** — attention بهتر برای classification
4. **Compression recovery** — کاربرد مهم
5. **Multi-architecture** — ۴ مدل

### نکته مهم در `variables.json`

- `LORA_RANK = 16` (پیش‌فرض)
- `LORA_ALPHA = 32` (2× rank)
- اگر rank را ۲ برابر کنی، alpha را هم ۲ برابر کن

### هشدارها

1. **`copy_weights` هنوز `raise` می‌کند** — باید fix شود
2. **`inject_lora_attention` هنوز `raise` می‌کند** — باید fix شود
3. **کد تکراری در `inject_lora_attention`** — باید تمیز شود

اگر می‌خواهی این سه باگ fix شوند، من می‌توانم patch رو بنویسم.

---

**قدم بعدی: `run_all_4_lora_attention.py` را با `--lora_rank 16 --lora_alpha 32` اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**