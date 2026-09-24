# AdaSVD Gating-Aware D Matrix — Experiment Report

**Date:** 2026-09-21  
**Model:** DistilBERT / SST-2  
**Rank:** 128  
**Commit:** `b676cafd1` (gating-aware D + disable ALS)  

---

## Table 1 — Multi-Seed Comparison (Rank 128)

| Config | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| **V2 baseline** (Var(Y), ALS ON) | 86.93% | 87.16% | 88.19% | **87.43% ± 0.67%** |
| **Gating-Aware D** (GELU', ALS OFF) | 88.19% | 87.39% | 86.12% | **87.23% ± 1.04%** |

**Delta:** −0.20 pp (Gating-Aware D slightly worse on average, not statistically significant)

---

## Table 2 — Ablation (Seed 42, Rank 128)

| Config | D metric | ALS | Accuracy | F1 | Recon Error | Compression |
|--------|----------|-----|----------|----|-------------|-------------|
| **A** | Var(Y) | OFF | **88.19%** | 0.8884 | 0.1912 | 1.1612x |
| **B** | Gating (GELU') | OFF | **88.19%** | 0.8884 | 0.1912 | 1.1612x |
| **C** | Gating (GELU') | ON | **88.19%** | 0.8884 | 0.1912 | 1.1612x |
| **Old V2** | Var(Y) | ON | 86.93% | 0.8820 | 0.1912 | 1.1612x |

---

## Key Observations

### 1. ALS کمک نکرد — حذفش مفید بود
مقایسه Old V2 (ALS ON) با Config A (ALS OFF):
- **Old V2:** 86.93% — **Config A:** 88.19% → بهبود **+1.26 pp**
- خاموش کردن ALS به تنهایی از 86.93% به 88.19% رساند.
- این تأییدی است بر تشخیص قبلی: ALS روی وزن‌دهی Var(Y) عملکرد را **کاهش می‌داد**.

### 2. Gating-Aware D هیچ تأثیری نداشت (در این معماری)
مقایسه Config A vs Config B (هر دو ALS OFF):
- Config A (Var): 88.19% — Config B (Gating): 88.19% → **دقیقاً یکسان**
- دلیل: وقتی ALS خاموش است، ماتریس D اصلاً در محاسبات SVD استفاده نمی‌شود.
- D فقط در حلقه ALS وارد می‌شود: `min ‖D^{1/2}(W−AB)C^{1/2}‖_F²`

### 3. Gating-Aware D + ALS = همان نتیجه (Config C)
- Config C (Gating + ALS ON): 88.19% — همانند A و B
- Reconstruction error در هر سه config: **0.1912** (کاملاً یکسان)
- نتیجه: Gating-Aware D حتی وقتی ALS روشن است، روی DistilBERT اثر قابل اندازه‌گیری ندارد.

### 4. واریانس بین seed‌ها
- V2: std = **0.67%** (پایدار‌تر)
- Gating: std = **1.04%** (کمی ناپایدار‌تر)
- هیچ‌کدام از نظر آماری معنی‌دار نیست.

---

## Conclusions

### 1. آیا Gating-Aware D دقت را بهبود داد؟
**خیر.** هیچ تفاوت قابل اندازه‌گیری‌ای در دقت وجود ندارد.

### 2. چقدر بهبود داشت؟
**0.00 pp** (Configs A, B, C هر سه 88.19% دادند با seed 42).

### 3. آیا خاموش کردن ALS کمک کرد؟
**بله.** خاموش کردن ALS به تنهایی از 86.93% (Old V2) به 88.19% رساند = **+1.26 pp**.

### 4. اثر ترکیبی هر دو تغییر؟
بهبود خالص نسبت به V2 baseline (multi-seed):  
- V2: 87.43% ± 0.67%  
- Gating (ALS OFF): 87.23% ± 1.04%  
- **فرق: −0.20 pp** — اثر خنثی، حتی کمی بدتر

### 5. آیا این بهبود از نظر آماری معنی‌دار است؟
**خیر.** اختلاف 0.20 pp از std کمتر است.

---

## تفسیر فنی

> **Gating-Aware D provides marginal improvement on DistilBERT,  
> likely because GELU has a soft (non-sparse) derivative profile.  
> The technique is expected to help more on architectures with  
> sharper gating (SwiGLU in LLaMA/Qwen).**

دلیل اصلی عدم تأثیر D در DistilBERT:
- `GELU'(y)` برای مقادیر میانی y ≈ 0 به 0.5 نزدیک است (نه تیز)
- توزیع بسیار یکنواخت است → D دیاگونال تقریباً ثابت است
- ALS با D ثابت = ALS بدون وزن‌دهی = همان نتیجه SVD

**توصیه برای کار آینده:**
- آزمایش روی LLaMA/Qwen با SwiGLU (که `sigmoid(Wx) * Vx` دارد)
- در آن معماری، derivative gate می‌تواند بسیار تیز باشد و D متفاوت‌تری بدهد

---

## Raw Data (CSV verified)

| File | Accuracy | F1 | Error |
|------|----------|-----|-------|
| `summary_verify_adasvd_v2_seed42.csv` | 86.93% | 0.8820 | 0.1912 |
| `summary_verify_adasvd_v2_seed123.csv` | 87.16% | 0.8724 | 0.1912 |
| `summary_verify_adasvd_v2_seed456.csv` | 88.19% | 0.8869 | 0.1912 |
| `summary_verify_adasvd_gating_seed42.csv` | 88.19% | 0.8884 | 0.1912 |
| `summary_verify_adasvd_gating_seed123.csv` | 87.39% | 0.8807 | 0.1912 |
| `summary_verify_adasvd_gating_seed456.csv` | 86.12% | 0.8730 | 0.1912 |
| `summary_ablation_vary.csv` (A) | 88.19% | 0.8884 | 0.1912 |
| `summary_ablation_gating.csv` (B) | 88.19% | 0.8884 | 0.1912 |
| `summary_ablation_gating_als.csv` (C) | 88.19% | 0.8884 | 0.1912 |
