# Pre-Submission Checklist — PHANTOM

## Setup
- [ ] Copied `_TEMPLATE/` to `phantom/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module phantom`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] All numbers match Section 7
- [ ] CUR and LSH both explained
- [ ] Softpick with fallback explained
- [ ] Parameter reduction verified
- [ ] Bucket size distribution reported

## Honesty
- [ ] Parameter reduction is real
- [ ] Accuracy delta honest
- [ ] Multi-round LSH overhead stated
- [ ] Bucket imbalance discussed
- [ ] Fallback frequency reported
- [ ] Single-seed limitation stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented
- [ ] LSH setup described

## Final
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/phantom
nano papers/phantom/variables.json
cd papers/phantom
python apply_vars.py --module phantom
grep -rn "{{" .  # empty
python build.py --module phantom --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE phantom
```

### مرحله ۲: اجرای PHANTOM

```powershell
cd D:\SVD_Project
python run_all_4_phantom.py --models distilbert,roberta,gpt2 --cur_rank 16 --lsh_rounds 4 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_phantom/*/summary.csv`.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\phantom
python apply_vars.py --module phantom
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module phantom
```

---

## 💡 نکات کلیدی PHANTOM

### ۱. سه محور فشرده‌سازی

PHANTOM یک ماژول **hybrid** است:
- **CUR**: پارامتر پروجکشن‌ها را کم می‌کند
- **LSH**: زمان attention را از $O(L^2)$ به $O(L \cdot B)$ می‌کند
- **Softpick**: sparsity بیشتری به attention می‌دهد

این سه را در abstract و introduction برجسته کن.

### ۲. rank=16 خیلی تهاجمی است

با `cur_rank=16`، کاهش **15×** در پروجکشن‌ها داری. این خیلی زیاد است ولی ممکن 
است accuracy افت کند.

اگر خواستی accuracy بهتر، `cur_rank=32` یا `cur_rank=64` بده.

### ۳. LSH bucket size

`n_bits=4` یعنی ۱۶ bucket. با $L=128$، میانگین هر bucket ~۸ توکن.

### ۴. Softpick fallback

fallback وقتی sum صفر است. باید گزارش کنی چه درصدی از rowها fallback می‌شوند.

### ۵. تفاوت با A3 و AASVD

| ویژگی | A3 | AASVD | PHANTOM |
|---|---|---|---|
| Projection | SVD + Fisher | SVD + cov | CUR + leverage |
| Attention | Dense | Dense | **LSH sparse** |
| Activation | Softmax | Softmax | **Softpick** |
| Rank default | 64 | 288 | **16** |

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_phantom/*/summary.csv` پر کن. اگر خواستی، من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**