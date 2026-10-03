# Pre-Submission Checklist — Gated DeltaNet

## Setup
- [ ] Copied `_TEMPLATE/` to `gated_delta_net/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module gated_delta_net`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Honest Reporting
- [ ] Abstract states the implementation is slow
- [ ] Abstract states RoBERTa/GPT-2 OOM
- [ ] Section 7 reports timeouts and OOMs honestly
- [ ] Section 8 fully explains the Python loop bottleneck
- [ ] Section 10.1 lists implementation as primary limitation
- [ ] Conclusion distinguishes algorithm from implementation

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Both algorithm (positive) and implementation (negative) covered
- [ ] Honest about OOM
- [ ] Honest about slow timing

### Introduction
- [ ] Motivates linear attention
- [ ] Formalizes the delta rule
- [ ] Contributions honestly framed

### Method
- [ ] Full derivation of delta rule
- [ ] Gating mechanism explained
- [ ] Bidirectional scan described
- [ ] Complexity analysis distinguishes FLOPs from wall-clock

### Results
- [ ] All tables filled
- [ ] Timing measured per-step
- [ ] OOM explicitly reported
- [ ] Loss curves shown

### Analysis
- [ ] Three-factor breakdown of slowness
- [ ] Comparison with published Triton implementations
- [ ] Recommendations for fixing

## Reproducibility
- [ ] Code on GitHub
- [ ] Timing scripts included
- [ ] Seeds documented
- [ ] OOM conditions documented

## Appendices
- [ ] A: Mathematical derivation
- [ ] B: Implementation details
- [ ] C: Full data including timing
- [ ] D: Complete negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render
- [ ] Consistent style

## Final Pass
- [ ] Read aloud
- [ ] Check every number
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (consider TMLR for negative results)
- [ ] Format per guidelines
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE gated_delta_net
```

### مرحله ۲: بررسی نتایج قبلی

از لاگ‌های تو:
- DistilBERT: timeout، per-step ~۹۰ ثانیه
- RoBERTa: OOM
- GPT-2: OOM

### مرحله ۳: پر کردن `variables.json` (بالا)

### مرحله ۴: Apply و build

```powershell
cd papers\gated_delta_net
python apply_vars.py --module gated_delta_net
python build.py --module gated_delta_net
```

---

## ⚠️ نکات مخصوص Gated DeltaNet

### ۱. این ماژول عملاً کاربردی نیست

- **۹۰ ثانیه per 5 steps** روی DistilBERT
- **OOM** روی RoBERTa و GPT-2
- **بدون Triton/CUDA kernel** نمی‌شود از آن استفاده کرد

مقاله باید این رو **شفاف** بگه.

### ۲. تفاوت algorithm vs implementation

**نکته مهم:** الگوریتم Gated DeltaNet درست است. نتایج بد از **پیاده‌سازی Python** می‌آید. مقایسه با Mamba/GLA که Triton kernel دارند ناعادلانه است.

### ۳. دلایل شکست

1. **Python loops:** ~۶۰۰ kernel launch در هر forward
2. **Gradient checkpointing:** ۲× محاسبه
3. **State memory:** $O(d^2 H)$ بزرگ
4. **بدون fused kernel**

### ۴. اگر بعداً Triton بنویسی

این تمپلیت رو نگه دار. وقتی kernel رو نوشتی:
- `D_STEP_TIME` رو به عدد جدید تغییر بده
- `D_STATUS` رو از `TIMEOUT` به `PASS` ببر
- بخش Section 8 (تحلیل کندی) رو با اعداد جدید به‌روز کن
- عنوان مقاله رو از «implementation bottleneck» به «fused kernel» تغییر بده

### ۵. Venue مناسب

مقالات با negative result + analysis در:
- **TMLR** (Transactions on Machine Learning Research)
- **arXiv**
- **Workshops** (NeurIPS efficiency workshop)

### ۶. درس اصلی برای community

- **Linear ≠ Fast.** O(L) با پیاده‌سازی ضعیف می‌تونه O(L²) رو ببره.
- **Fused kernels اجباری هستند.**
- **Benchmark باید wall-clock رو اندازه بگیره، نه FLOPs.**

---

**قدم بعدی: اگر خواستی می‌تونم `variables.json` رو با اعداد دقیق‌تر از لاگ‌های خودت پر کنم. لاگ‌های بنچمارک این ماژول رو بفرست.**