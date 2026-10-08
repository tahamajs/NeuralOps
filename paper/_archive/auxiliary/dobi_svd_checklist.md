# Pre-Submission Checklist — Dobi-SVD

## Setup
- [ ] Copied `_TEMPLATE/` to `dobi/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module dobi`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Differentiable rank selection emphasized**
- [ ] **Learnable spectral gate highlighted**
- [ ] Gate compensation mentioned
- [ ] Post-hoc pruning mentioned

## Introduction
- [ ] Motivation: discrete rank problem
- [ ] Problem statement formal
- [ ] Three design choices clear
- [ ] Contributions verifiable

## Method
- [ ] All formulas numbered
- [ ] `DobiSVDLinearLayer` described
- [ ] Gate compensation proven
- [ ] Post-hoc pruning algorithm
- [ ] Complexity analysis

## Architecture
- [ ] Four models supported
- [ ] LoRALinear wrapping handled
- [ ] Conv1D handled
- [ ] Registry entries documented

## Results
- [ ] Table 7.1 (main) filled
- [ ] Table 7.2 (effective rank) — unique
- [ ] Table 7.6 (rank cap)
- [ ] Table 7.7 (init gate) — unique
- [ ] Table 7.8 (pruning) — unique
- [ ] Table 7.11 (gate distribution) — unique

## Honesty
- [ ] Real parameter reduction verified
- [ ] Effective rank is smaller than nominal
- [ ] Post-hoc pruning drop reported
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Post-hoc pruning script included
- [ ] Gate dynamics documented

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/dobi
nano papers/dobi/variables.json
cd papers/dobi
python apply_vars.py --module dobi
grep -rn "{{" .  # should be empty
python build.py --module dobi --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE dobi
```

### مرحله ۲: اجرای Dobi-SVD

```powershell
cd D:\SVD_Project
python run_all_4_dobi.py --models distilbert,roberta,gpt2 --rank 128 --init_gate_val 2.5
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_dobi/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\dobi
python apply_vars.py --module dobi
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module dobi
```

---

## 💡 نکات مخصوص Dobi-SVD

### ۱. **Learnable Spectral Gate** — نوآوری اصلی

این تنها روشی است که rank را **differentiable** می‌کند. در abstract، introduction، و method برجسته کن.

فرمول کلیدی:
$$
y = (x (V \odot \sigma(\alpha))^\top) U^\top + b
$$

### ۲. **Gate Compensation** — نکته حیاتی

`init_gate_scale = 1/σ(α_init)` — این باعث می‌شود که در step 0 بازسازی **دقیقاً** با SVD مطابق باشد.

### ۳. **Post-Hoc Pruning** — مزیت منحصربفرد

بعد از training:
```python
r_eff = sum(sigmoid(gate) > 0.5)
```

این را می‌توان بدون retraining انجام داد.

### ۴. **Gate Distribution Analysis** — بخش یکتا

Table 7.11 و Section 8.2 **فقط** در این مقاله معنی دارند چون فقط این روش gate دارد.

### ۵. **تفاوت با AdaLoRA**

AdaLoRA **budget** learns, Dobi-SVD **rank components** learns. در related work توضیح بده.

### ۶. **Reconstruction Error Definition**

در `compress_linear`، خطای بازسازی از **پس از gate compensation** محاسبه می‌شود. این مهم است.

---

**قدم بعدی: `run_all_4_dobi.py` را اجرا کن. نتایج را از `summary.csv` بگیر و در `variables.json` بگذار. اگر CSV را بفرستی، من می‌توانم `variables.json` را با اعداد واقعی پر کنم.**