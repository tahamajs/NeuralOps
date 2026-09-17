# Pre-Submission Checklist — ExpertChoiceMoEAttention

## Setup
- [ ] Copied `_TEMPLATE/` to `expert_choice_moe/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module expert_choice_moe`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Expert-choice routing claim stated**
- [ ] **Guaranteed load balancing claim stated**
- [ ] **Parameter increase stated explicitly**
- [ ] Zero-init stability mentioned
- [ ] Cross-attention support mentioned

## Intro
- [ ] Motivation for expert-choice (vs token-choice)
- [ ] Problem statement formal
- [ ] Contributions listed
- [ ] No auxiliary loss advantage highlighted

## Method
- [ ] Expert-choice routing formula
- [ ] Zero-init guarantee proof sketch
- [ ] Gate initialization
- [ ] **Load balance proof**
- [ ] Parameter analysis **honest**
- [ ] Dynamic router explained

## Results
- [ ] Main accuracy table
- [ ] Parameter table (honest)
- [ ] **Expert utilization table (perfectly balanced)**
- [ ] Expert count ablation
- [ ] Capacity factor ablation
- [ ] Gate initialization ablation
- [ ] Zero-init ablation
- [ ] Comparison to token-choice (BAM-UMoE)

## Honesty
- [ ] **Parameter increase (+48.5\%%) stated in abstract, results, discussion**
- [ ] "Modest accuracy gain" stated
- [ ] Python loop limitation stated
- [ ] Cross-attention not fully tested
- [ ] Single-seed stated
- [ ] Not deployment-ready stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Expert count documented
- [ ] Capacity factor documented
- [ ] Batch_size=4 note for OOM

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/expert_choice_moe
nano papers/expert_choice_moe/variables.json
cd papers/expert_choice_moe
python apply_vars.py --module expert_choice_moe
grep -rn "{{" .
python build.py --module expert_choice_moe --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **نوآوری اصلی: Expert-Choice Routing Inside Attention**

این اولین ماژولیه که **expert-choice routing** رو داخل attention استفاده می‌کنه. تفاوت با BAM-UMoE:
- BAM-UMoE: token → expert (token-choice)
- ExpertChoiceMoE: expert → token (expert-choice)

### ۲. **مزیت کلیدی: Perfect Load Balancing**

برخلاف token-choice که نیاز به aux loss داره و بازم imbalance داره، expert-choice **تضمین** می‌کنه که هر expert دقیقاً k token بگیره.

### ۳. **صداقت درباره پارامتر**

+807% پارامتر اضافه می‌کنه — بزرگ‌ترین limitation. باید صریحاً بگی.

### ۴. **Python Loop**

حلقه `for e in range(n_experts)` سرعت رو محدود می‌کنه. این رو در limitations باید بگی.

### ۵. **تست سریع**

```bash
# با batch_size=4 برای جلوگیری از OOM
python run_all_4_expert_choice_moe.py --models distilbert --n_experts 8 --capacity_factor 1.0 --batch_size 4
```

اگر OOM داد، `--n_experts 4` رو امتحان کن.

---

**قدم بعدی: `run_all_4_expert_choice_moe.py` را اجرا کن، اعداد را از `summary.csv` و لاگ auxiliary loss بردار، در `variables.json` بگذار.**