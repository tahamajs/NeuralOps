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
```

---

## 💡 نکات کلیدی Squeezing Heads

### ۱. تفاوت اصلی با سایر ماژول‌ها

| ماژول | هدف | پارامتر اضافه |
|---|---|---|
| ASA-Sparse | sparse attention | sparsity_logits |
| NHA | hybrid attention | gate |
| NSA | local+topk | هیچ |
| **SH** | **head distillation** | **LoRA** |

### ۲. Wrapper، نه جایگزین

SH **attention اصلی را دست نمی‌زند** — فقط LoRA اضافه می‌کند. Forward اول دقیقاً برابر baseline است.

### ۳. Head-Dim Reduction

$$
W_Q^{(\text{new})} = W_Q^{(\text{old})}_{:d_h' H, :}
$$

حفظ **leading dimensions** (نه random slice).

### ۴. GQA Pooling

وقتی source heads بیشتر از target است:

$$
W_K^{(\text{pool})}_g = \frac{1}{r} \sum_{h \in g} W_K^{(\text{source})}_h
$$

### ۵. LoRA Recovery — نکته کلیدی

- **بدون LoRA:** head-dim reduction → افت شدید
- **با LoRA:** افت کم
- **Gain:** {{LORA_GAIN}}%

### ۶. Trade-off مهم

| فاکتور | Base | با head-dim 2× |
|---|---|---|
| Params | 100% | ~50% |
| Trainable | 100% | ~1% (LoRA) |
| Inference | پایه | **کمی کندتر** |
| Storage | 100% | 100% (base+LoRA) |

**صادقانه:** SH **سرعت inference را کم نمی‌کند** مگر LoRA را در base fold کنی.

### ۷. نتایج از لاگ

از لاگ‌های قبلی:
- **DistilBERT:** ~90.25%
- **RoBERTa:** ~76.8%
- **GPT-2:** ~77.5%

این‌ها تخمینی هستند — اگر اجرا کرده‌ای، اعداد واقعی را وارد کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_squeezing_heads/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**