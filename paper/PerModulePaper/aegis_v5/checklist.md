# Pre-Submission Checklist — AEGIS-V5

## Setup
- [ ] Copied `_TEMPLATE/` to `aegis_v5/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module aegis_v5`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Split-head mentioned
- [ ] SVD + LoRA mentioned
- [ ] Softpick fix highlighted
- [ ] Parameter effect stated

## Method
- [ ] SVD formula present
- [ ] Energy rank formula
- [ ] Split-head equation
- [ ] Gate equation
- [ ] Causal fallback explained

## Results
- [ ] Main results (Table 7.1)
- [ ] Parameter analysis (Table 7.2)
- [ ] Rank ablation
- [ ] Head split ablation
- [ ] Window ablation
- [ ] Softpick V4 vs V5

## Honesty
- [ ] Parameter effect honest
- [ ] Decoder fallback limitation stated
- [ ] Single-seed limitation
- [ ] Failure modes documented

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/aegis_v5
nano papers/aegis_v5/variables.json
cd papers/aegis_v5
python apply_vars.py --module aegis_v5
grep -rn "{{" .
python build.py --module aegis_v5 --pdf
```
```

---

## 🎯 خلاصه نهایی

### ویژگی‌های منحصر به فرد AEGIS-V5

1. **Split-head architecture** — ۲/۳ local + ۱/۳ global
2. **Energy-based SVD** — انتخاب rank با آستانه ۰.۹۵
3. **Stable softpick** — fix از ReLU-based به softmax
4. **Safe copy_weights** — warning به جای raise
5. **Decoder fallback** — dense causal برای صحت
6. **Per-head adaptive gate** — با init bias 2.0

### مقایسه با مقالات قبلی

| مقاله | نوآوری اصلی | پارامتر | Complexity |
|---|---|---|---|
| NHA | Hybrid local+global | + | $\mathcal{O}(L^2)$ |
| LEKAv3 | Landmarks + local | + | $\mathcal{O}(L \cdot L_\text{land})$ |
| Linformer | Sequence projection | + | $\mathcal{O}(L \cdot k)$ |
| **AEGIS-V5** | **Split-head + SVD** | **depends** | **$\mathcal{O}(L(W+F))$** |

### برجسته کردن در مقاله

1. **Softpick fix** — یک fix ساده ولی حیاتی
2. **Split-head** — expressiveness
3. **Energy-based SVD** — کنترل explicit روی compression
4. **Safe fallback** — production-hardening

### نکته مهم در `variables.json`

- `MAX_RANK = 384` (default) → **هیچ compression**
- `MAX_RANK = 128` → compression ~۱.۵×
- `MAX_RANK = 64` → compression ~۲.۵×

اگر می‌خواهی compression واقعی داشته باشی، `MAX_RANK` را کم کن.

---

**قدم بعدی: `run_all_4_aegis_v5.py` را با `--max_rank 128` اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**