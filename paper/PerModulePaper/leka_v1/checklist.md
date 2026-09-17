# Pre-Submission Checklist — LEKAv1

## Setup
- [ ] Copied `_TEMPLATE/` to `leka_v1/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module leka_v1`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Linear-time claim
- [ ] RFF approach stated
- [ ] Parameter increase stated
- [ ] Not a compression method

## Intro
- [ ] Linear attention motivation
- [ ] RFF explained
- [ ] Contributions

## Method
- [ ] Feature map formula
- [ ] QR init
- [ ] Preservation gate
- [ ] Complexity analysis
- [ ] Parameter analysis (honest)

## Results
- [ ] Main accuracy
- [ ] Parameter table
- [ ] Speed vs L table
- [ ] Feature dim ablation
- [ ] Gate ablation
- [ ] LN ablation
- [ ] QR ablation
- [ ] LoRA ablation

## Honesty
- [ ] Parameter increase stated
- [ ] Cumsum memory noted
- [ ] Fatal error handling noted
- [ ] Single-seed stated
- [ ] Cross-attention not supported

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Feature dim documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/leka_v1
nano papers/leka_v1/variables.json
cd papers/leka_v1
python apply_vars.py --module leka_v1
grep -rn "{{" .
python build.py --module leka_v1 --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **ماهیت: Linear Attention واقعی**

این ماژول O(L·F) هست، نه O(L²). این را در:
- Abstract
- Intro
- Method (با math کامل)
- Discussion

برجسته کن.

### ۲. **RFF برای Gaussian Kernel**

فرق مهم با Performer:
- Performer: PORF برای **softmax kernel** ($\exp(q \cdot k)$)
- LEKAv1: RFF برای **Gaussian kernel** ($\exp(-\|q-k\|²/2)$)

### ۳. **سه نوآوری واقعی**

1. **Head-wise LayerNorm قبل از projection**
2. **QR-orthogonalized projection**
3. **Preservation gate** ($\alpha$ mixing)

### ۴. **صداقت درباره پارامتر**

پارامتر **۲.۷٪ بیشتر** می‌شه. این را واضح بگو.

### ۵. **دو باگ که باید fix بشن**

1. `raise RuntimeError` در `copy_weights` (کشنده)
2. `output_scale` پارامتر جدید که در checkpoint نیست

### ۶. **Cumsum memory**

برای L > 512، cumsum materialization می‌تونه OOM بده. این را در limitations برجسته کن.

### ۷. **تست سریع**

```bash
python run_all_4_leka_v1.py --models distilbert,roberta,gpt2 --feature_dim 256 --batch_size 4
```

---

**قدم بعدی: `run_all_4_leka_v1.py` را اجرا کن، اعداد را از `summary.csv` بگیر، در `variables.json` بگذار.**