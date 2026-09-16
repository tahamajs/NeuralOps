# Pre-Submission Checklist — BAM-UMoE

## CRITICAL
- [ ] **Never call BAM-UMoE "compression"**
- [ ] Parameter increase clearly stated (~350%)
- [ ] Step-0 preservation verified (diff < 1e-5)
- [ ] Cross-attention scope documented
- [ ] Honest cost-benefit analysis

## Content
- [ ] Abstract under 250 words
- [ ] Step-0 mechanism clearly explained
- [ ] Load balancing loss documented
- [ ] All ablations (E, k, gate bias) present
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Step-0 verification test provided
- [ ] Expert utilization metrics included

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/bam_umoe
nano papers/bam_umoe/variables.json
cd papers/bam_umoe
python apply_vars.py --module bam_umoe
grep -rn "{{" .
python build.py --module bam_umoe --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE bam_umoe
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_bam_umoe.py --models distilbert,roberta,gpt2 --num_experts 8 --top_k 2 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\bam_umoe
python apply_vars.py --module bam_umoe
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module bam_umoe --pdf
```

---

## 💡 نکات حیاتی BAM-UMoE

### ۱. هرگز نگو "Compression"

این ماژول **~350% پارامتر اضافه می‌کند**. اگر در مقاله «compression» بنویسی، 
**داده تقلبی** است.

### ۲. Step-0 Preservation — نوآوری اصلی

سه‌گانه:
1. Zero-init final expert layer
2. Gate bias = -3.0 (sigmoid ≈ 0.047)
3. Frozen base

این را در abstract برجسته کن.

### ۳. Load Balancing Loss

این برای MoE حیاتی است. در method و analysis توضیح بده.

### ۴. Cross-Attention Skip

MoE روی cross-attention اعمال نمی‌شود — دلیلش حفظ alignment است. این را 
صادقانه بگو.

### ۵. مقایسه با LoRA تنها

اگر BAM-UMoE دقت مشابه LoRA می‌دهد ولی ۴.۴× پارامتر، این **trade-off بد** 
است. صادقانه گزارش کن.

### ۶. Where it shines

اگر BAM-UMoE روی **multi-task** یا **capacity-limited** setting بهتر عمل 
کند، آن را برجسته کن. روی single-task SST-2 احتمالاً ارزشش کمتر است.

---

**قدم بعدی: `run_all_4_bam_umoe.py` را اجرا کن و نتایج را بفرست تا 
`variables.json` را با اعداد واقعی پر کنم.**