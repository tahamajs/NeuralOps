# Pre-Submission Checklist — SVD-Linear Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `svd_linear_attention/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module svd_linear_attention`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Real parameter reduction stated**
- [ ] **Real complexity reduction stated**
- [ ] Streaming capability mentioned
- [ ] Decay gating mentioned

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this IS a compressor**
- [ ] Contributions verifiable

### Method
- [ ] All formulas numbered
- [ ] Linear attention identity derived
- [ ] Chunked scan described
- [ ] Decay gating explained
- [ ] Streaming mode described
- [ ] Complexity analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows reduction**
- [ ] Rank ablation
- [ ] Mode ablation
- [ ] Decay ablation
- [ ] Streaming results
- [ ] Scaling with N

### Analysis
- [ ] Why it works
- [ ] Per-layer analysis
- [ ] Decay distribution
- [ ] Comparison with AASVD
- [ ] Comparison with RPC

### Limitations
- [ ] Approximation error
- [ ] Task coverage
- [ ] Cross-attention not supported
- [ ] Decay init bug documented
- [ ] Single-seed

## Honesty
- [ ] Real parameter reduction
- [ ] Real speedup
- [ ] Decay init bug documented
- [ ] Failure cases documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Streaming example provided

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/svd_linear_attention
nano papers/svd_linear_attention/variables.json
cd papers/svd_linear_attention
python apply_vars.py --module svd_linear_attention
grep -rn "{{" .
python build.py --module svd_linear_attention --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE svd_linear_attention
```

### مرحله ۲: اجرای SVD-Linear Attention

```powershell
cd D:\SVD_Project
python run_all_4_svd_linear_attention.py --models distilbert,roberta,gpt2 --rank 32 --mode decay_linear --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\svd_linear_attention
python apply_vars.py --module svd_linear_attention
python build.py --module svd_linear_attention
```

---

## 💡 نکات کلیدی SVD-Linear Attention

### ۱. **SVD-Linear Attention یک compressor واقعی است**

- SVD factorization → پارامتر **کم** می‌کند
- نسبت کاهش: **~50\%×** در `r=32`
- برخلاف NHA/EAT/OHA/Infini/SVD-Mamba که پارامتر **اضافه** می‌کنند

### ۲. **پیچیدگی خطی**

- Softmax: $\mathcal{O}(N^2)$
- Linear: $\mathcal{O}(N)$
- برای N=1024: **~15× speedup** (تئوری)

### ۳. **دو حالت اصلی**

- `exact`: softmax (بدون کاهش پیچیدگی)
- `linear` / `decay_linear`: kernelized (با کاهش پیچیدگی)

### ۴. **Decay gating**

- `λ = exp(-softplus(W_g x))` ∈ (0, 1)
- **Log-spaced init** (ولی معکوس — باگ در A.6)
- Heads converge to different scales

### ۵. **Streaming**

- `step_recurrent` برای **O(1)** inference
- مناسب برای autoregressive generation
- مثال در §12.8

### ۶. **مشکلات**

- **Decay init inversion** (A.6) — bug
- **Cross-attention** not supported
- **Chunk size** fixed at 64
- No activation weighting (unlike AASVD)

### ۷. **مزیت بر AASVD**

| ویژگی | AASVD | SVD-Linear |
|---|---|---|
| پارامتر | **-50%** | **-[D_PARAM_PCT]%** |
| پیچیدگی | $\mathcal{O}(N^2)$ | **$\mathcal{O}(N)$** |
| Streaming | No | **Yes** |
| Weighting | Full covariance | None |

### ۸. **مزیت بر RPC**

RPC:
- پارامتر **اضافه** می‌کند
- حافظه **کم نمی‌کند**
- پیچیدگی $\mathcal{O}(N^2)$ باقی می‌ماند

SVD-Linear:
- پارامتر **کم** می‌کند
- حافظه **کم** می‌کند
- پیچیدگی **$\mathcal{O}(N)$**

**SVD-Linear به شدت بهتر است.**

---

**قدم بعدی: `run_all_4_svd_linear_attention.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**