# Pre-Submission Checklist — LEKAv3

## Setup
- [ ] Copied `_TEMPLATE/` to `leka_v3/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module leka_v3`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Complexity claim stated
- [ ] Learned landmarks emphasized
- [ ] Local branch mentioned
- [ ] Parameter increase honest

## Method
- [ ] Feature map formula
- [ ] Landmark scorer formula
- [ ] Fusion equation
- [ ] Causality proof
- [ ] Complexity analysis

## Results
- [ ] Main results (Table 7.1)
- [ ] Parameter count (Table 7.2)
- [ ] $L_0$ ablation
- [ ] $W$ ablation
- [ ] $d_\text{feat}$ ablation
- [ ] Component ablation
- [ ] Gate init ablation
- [ ] Scaling table

## Honesty
- [ ] Parameter increase stated
- [ ] No overclaiming
- [ ] Failure modes documented
- [ ] Single-seed limitation

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/leka_v3
nano papers/leka_v3/variables.json
cd papers/leka_v3
python apply_vars.py --module leka_v3
grep -rn "{{" .
python build.py --module leka_v3 --pdf
```
```

---

## 🚀 خلاصه

### ویژگی‌های منحصر به فرد LEKAv3

1. **Landmark scoring روی K-features** (نه hidden states)
2. **Multi-scale RFF** با دو پروجکشن
3. **Learnable gate** با init ۲.۰
4. **W_corr** matrix برای mixing per-head landmarks
5. **softpick** به جای softmax (sparsifying)
6. **Adaptive landmark count:** `min(N, max(L0, N//2))`

### تفاوت با NHA/NASC

- NHA: ۵ kernel launch، ۵x کندتر → **fail**
- NASC: ۲۴ لایه SVD، delta ≈ ۰ → **no-op روی DistilBERT**
- LEKAv3: **linearly scalable، ۵ نوآوری، پارامتر +۷.۵٪**

### نکته مهم در `variables.json`

- `L0 = 64` (پایه)
- `W = 16` (پنجره)
- `FEATURE_DIM = 128`

این‌ها را می‌توانی tune کنی، ولی این‌ها مقادیر پیشنهادی هستند.

### در مقاله برجسته کن

1. **Learned landmark selection** — اولین‌بار که landmark scorer با LoRA train می‌شود
2. **Linear complexity** — اثبات ریاضی
3. **Multi-scale features** — یک نوآوری کوچک
4. **Gate init 2.0** — اهمیت initialization

---

**قدم بعدی: `run_all_4_leka_v3.py` را اجرا کن با `--num_landmarks 64 --local_window_size 16`، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**