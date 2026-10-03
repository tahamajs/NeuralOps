# Pre-Submission Checklist — HGRN2Layer

## Setup
- [ ] Copied `_TEMPLATE/` to `hgrn2_layer/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module hgrn2_layer`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Linear complexity stated
- [ ] **Parameter increase stated explicitly**
- [ ] Short-sequence slowdown stated
- [ ] Breakeven L mentioned

## Intro
- [ ] Motivation clear
- [ ] Two design choices highlighted (bounded gate + input modulation)
- [ ] Contributions listed

## Method
- [ ] Forget gate formula
- [ ] Input gate formula
- [ ] Chunked scan explained
- [ ] Bidirectional support
- [ ] Parameter analysis honest

## Results
- [ ] Accuracy table
- [ ] Parameter table (honest)
- [ ] Speed vs L table
- [ ] Chunk ablation
- [ ] Bidirectional ablation
- [ ] Recovery ablation

## Honesty
- [ ] Parameter increase stated in abstract, results, discussion
- [ ] Short-sequence slowdown stated
- [ ] Python loop limitation stated
- [ ] Nested handling caveat
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Chunk size documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/hgrn2_layer
nano papers/hgrn2_layer/variables.json
cd papers/hgrn2_layer
python apply_vars.py --module hgrn2_layer
grep -rn "{{" .
python build.py --module hgrn2_layer --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **صداقت درباره پارامتر**

این ماژول **پارامتر کم نمی‌کند** — بلکه **۲۵٪ بیشتر** می‌کند. باید در:
- Abstract
- Results
- Discussion
- Limitations

این را برجسته کنی.

### ۲. **سرعت در توالی‌های کوتاه**

SST-2 با L=128 کار می‌کند که **دقیقاً** ناحیه کند این ماژول است. باید breakeven را دقیق محاسبه کنی (احتمالاً L* ≈ 512).

### ۳. **سه نوآوری واقعی**

1. **Bounded log-space forget gate** — جلوگیری از saturation
2. **Input modulation با (1-f)** — جلوگیری از drift
3. **Chunked parallel scan در FP32** — سرعت + پایداری

### ۴. **Bidirectional برای encoder**

DistilBERT و RoBERTa از bidirectional استفاده می‌کنند. GPT-2 از causal.

### ۵. **تفاوت با HGRN2GatedDeltaNet**

این ماژول **سبک‌تره** — فقط HGRN-2، بدون DeltaNet و gate. استفاده‌اش زمانی است که می‌خواهی سریع و ساده باشه.

---

**قدم بعدی: `run_all_4_hgrn2.py` را اجرا کن (اگر OOM داد، `--batch_size 4`)، اعداد را از `summary.csv` بگیر و در `variables.json` بگذار.**