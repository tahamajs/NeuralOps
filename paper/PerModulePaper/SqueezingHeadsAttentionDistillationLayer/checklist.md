# Pre-Submission Checklist — Squeezing Heads

## Setup
- [ ] Copied `_TEMPLATE/` to `squeezing_heads/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module squeezing_heads`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] LoRA wrapper described
- [ ] Head-dim reduction described
- [ ] GQA pooling described

## Method
- [ ] LoRA formula numbered
- [ ] Head-dim slicing formula
- [ ] Copy procedure algorithm
- [ ] Complexity table

## Results
- [ ] Main table
- [ ] Head-dim reduction ablation
- [ ] LoRA rank ablation
- [ ] Recovery steps ablation
- [ ] Comparison with pruning

## Honesty
- [ ] Inference slowdown noted
- [ ] Storage overhead noted
- [ ] Head-dim reduction loss noted
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final
- [ ] Read aloud
- [ ] PDF compiles
- [ ] External feedback

---

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/squeezing_heads
nano papers/squeezing_heads/variables.json
cd papers/squeezing_heads
python apply_vars.py --module squeezing_heads
grep -rn "{{" .  # verify empty
python build.py --module squeezing_heads --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE squeezing_heads
```

### مرحله ۲: اجرای Squeezing Heads

```powershell
cd D:\SVD_Project
python run_all_4_squeezing_heads.py --models distilbert,roberta,gpt2 --lora_rank 16 --head_dim_reduction 1.0 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_squeezing_heads/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\squeezing_heads
python apply_vars.py --module squeezing_heads
python build.py --module squeezing_heads
