# Pre-Submission Checklist — Adaptive Sparse Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_sparse/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_sparse`
- [ ] No `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] Learned per-head sparsity is clear
- [ ] Soft top-k explained
- [ ] Chunked memory saving stated

## Method
- [ ] Sparsity logits formula numbered
- [ ] Soft mask formula numbered
- [ ] Chunked complexity table complete
- [ ] Gradient flow explained

## Results
- [ ] Main table
- [ ] Sparsity ablation
- [ ] Temperature ablation
- [ ] Head budget distribution
- [ ] Attention mass retention

## Honesty
- [ ] No parameter reduction — stated
- [ ] No time reduction — stated
- [ ] Causality approximation — noted
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
cp -r papers/_TEMPLATE papers/adaptive_sparse
nano papers/adaptive_sparse/variables.json
cd papers/adaptive_sparse
python apply_vars.py --module adaptive_sparse
grep -rn "{{" .  # verify empty
python build.py --module adaptive_sparse --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_sparse
```

### مرحله ۲: اجرای ASA

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_sparse.py --models distilbert,roberta,gpt2 --sparsity_ratio 0.5 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adaptive_sparse
python apply_vars.py --module adaptive_sparse
python build.py --module adaptive_sparse
```

---

## 💡 نکات کلیدی ASA

### ۱. تفاوت اصلی با روش‌های دیگه

ASA **یاد می‌گیرد** که هر head چقدر باید sparse باشد.

پارامتر کلیدی: `sparsity_logits` — یک عدد learnable به ازای هر head.

### ۲. Soft Top-K

به جای hard top-k (که gradient ندارد):

$$
\tilde{s}_j = s_j + \log \sigma\left(\frac{s_j - \tau}{T}\right)
$$

این **قابل مشتق‌گیری** است.

### ۳. Head Budgets دو حالتی

یادگیری per-head sparsity نشان می‌دهد:
- **Critical heads:** sparsity کم (~0.1)
- **Redundant heads:** sparsity زیاد (~0.8)

این تفسیرپذیری از **مزایای اصلی** ASA است.

### ۴. بدون کاهش پارامتر یا زمان

ASA فقط **حافظه** را کم می‌کند. برای کاهش پارامتر از A3/AASVD و برای کاهش KV cache از GQA استفاده کن.

### ۵. Chunking = کاهش حافظه

بدون chunking، حتی با sparsity، باید $L \times L$ ماتریس ساخته شود. `chunk_size=64` این را به $C \times L$ کاهش می‌دهد.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_adaptive_sparse/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**