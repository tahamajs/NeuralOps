# Pre-Submission Checklist — CHELA-SLAY

## Setup
- [ ] Copied `_TEMPLATE/` to `chela_slay/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module chela_slay`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Hybrid claim (dense + linear) stated
- [ ] Spherical + Yat-Kernel mentioned
- [ ] Adaptive gate + init bias stated
- [ ] **Cumsum memory limitation stated**
- [ ] **Short-sequence slowdown stated**

## Intro
- [ ] Motivation for hybrid
- [ ] Three components listed
- [ ] Contributions verifiable

## Method
- [ ] Spherical normalization formula
- [ ] Yat-Kernel formula
- [ ] Linear attention formula
- [ ] Causal cumsum formula
- [ ] Gate fusion formula
- [ ] Step-0 preservation proof sketch
- [ ] **Complexity + memory analysis complete**

## Results
- [ ] Main accuracy table
- [ ] Parameter table
- [ ] Inference time table
- [ ] **Peak VRAM table**
- [ ] Feature dim ablation
- [ ] Gate bias ablation
- [ ] Stream ablation
- [ ] **Sequence-length scaling table**

## Honesty
- [ ] **Cumsum memory limitation stated in abstract, results, limitations**
- [ ] **Short-sequence slowdown stated**
- [ ] Gate remains dense-dominant reported
- [ ] Single-seed stated
- [ ] No theoretical bound stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Feature dim documented
- [ ] Gate bias documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/chela_slay
nano papers/chela_slay/variables.json
cd papers/chela_slay
python apply_vars.py --module chela_slay
grep -rn "{{" .
python build.py --module chela_slay --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **نوآوری سه‌گانه**

- **Spherical normalization**: Q, K را روی unit sphere می‌بره
- **Yat-Kernel**: cos+sin RFF
- **Adaptive gate**: dense + linear رو blend می‌کنه

### ۲. **محدودیت مهم: Cumsum Memory**

در causal mode، `torch.cumsum` یک tensor با شکل `(B, H, L, 2F, d_h)` می‌سازه. برای `L=128, 2F=256, d_h=64, B=8, H=12`: **~8 GB**. این بزرگ‌ترین محدودیت است.

### ۳. **Gate در ابتدا تقریباً 1.0**

با `b_g = 5.0`: `sigmoid(5) = 0.993`. یعنی در step 0 عملاً فقط dense اجرا می‌شه.

### ۴. **سرعت در L=128 کندتره**

چون **هر دو stream** محاسبه می‌شن. سرعت واقعی فقط در L بزرگ ظاهر می‌شه.

### ۵. **تست سریع**

```bash
# با batch_size=4 برای جلوگیری از OOM
python run_all_4_chela_slay.py --models distilbert,roberta,gpt2 --feature_dim 128 --batch_size 4
```

اگر OOM داد، `--feature_dim 64` رو امتحان کن.

---

**قدم بعدی: `run_all_4_chela_slay.py` را اجرا کن با `--batch_size 4 --feature_dim 64` برای شروع، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**