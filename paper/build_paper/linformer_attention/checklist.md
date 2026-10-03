# Pre-Submission Checklist — Linformer

## Setup
- [ ] Copied `_TEMPLATE/` to `linformer/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module linformer`
- [ ] No `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] Complexity stated
- [ ] Parameter increase honest
- [ ] Causal fallback explained
- [ ] Diagonal init explained

## Results
- [ ] Main results (Table 7.1)
- [ ] Parameter counts
- [ ] k ablation
- [ ] Init ablation
- [ ] Causal fallback ablation
- [ ] Scaling table

## Honesty
- [ ] Parameter increase stated
- [ ] Decoder limitation stated
- [ ] Failure modes documented
- [ ] Single-seed limitation

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/linformer
nano papers/linformer/variables.json
cd papers/linformer
python apply_vars.py --module linformer
grep -rn "{{" .
python build.py --module linformer --pdf
```
```

---

## 🎯 خلاصه نهایی

### ویژگی‌های منحصر به فرد این پیاده‌سازی Linformer

1. **Diagonal init E, F** → شروع از identity، همگرایی سریع
2. **Adaptive slicing** با ۳ حالت
3. **Causal fallback** برای decoder → صحت causality
4. **Multi-architecture:** GPT-2 Conv1D, RoBERTa nested, Qwen GQA+RoPE
5. **LoRA + output_scale** برای recovery

### تفاوت‌ها با مقالات قبلی

| مقاله | نوع | پارامتر | Complexity |
|---|---|---|---|
| NHA | attention جایگزین | + | $\mathcal{O}(L^2)$ |
| A3/AASVD | weight compression | - | $\mathcal{O}(L^2)$ |
| TCA | token compression | + | $\mathcal{O}(L \cdot k)$ |
| LEKAv3 | landmark + local | + | $\mathcal{O}(L \cdot L_\text{land})$ |
| **Linformer** | **seq projection** | **+۷٪** | **$\mathcal{O}(L \cdot k)$ encoder** |

### برجسته کردن در مقاله

1. **Causal-safe fallback** — یک نوآوری عملی مهم
2. **Adaptive slicing** — رفتار خودکار با طول‌های مختلف
3. **Diagonal init** — همگرایی سریع‌تر
4. **Multi-architecture** — ۴ مدل
5. **صداقت در پارامتر** — +۷٪ افزایش

### نکته مهم در `variables.json`

- `K = 64` (default)
- `MAX_SEQ = 512` (پیش‌فرض BERT)

اگر می‌خواهی فشرده‌سازی بیشتر، `K = 32` بگذار (ولی accuracy کم می‌شود).

---

**قدم بعدی: `run_all_4_linformer.py` را اجرا کن با `--linformer_k 64`، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**