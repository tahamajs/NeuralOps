# Pre-Submission Checklist — MoE-DeltaKV-Paged

## Setup
- [ ] Copied `_TEMPLATE/` to `moe_deltakv_paged/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module moe_deltakv_paged`
- [ ] No `[PLACEHOLDER]` remains

## Critical Honesty Checks
- [ ] **Delta-KV is simulation, stated in abstract**
- [ ] **4-bit quantization is simulation, stated**
- [ ] **No memory savings claim**
- [ ] **6D gather OOM risk documented**

## Abstract
- [ ] Under 250 words
- [ ] Three components mentioned
- [ ] Two caveats explicit
- [ ] CLS preservation mentioned

## Method
- [ ] Router formula
- [ ] Delta-KV pipeline
- [ ] STE formula
- [ ] CLS boost
- [ ] Aux loss

## Results
- [ ] Main results
- [ ] Router statistics
- [ ] k_experts ablation
- [ ] Page size ablation
- [ ] Component ablation
- [ ] Scaling

## Limitations
- [ ] Simulation caveat
- [ ] SVD-in-forward cost
- [ ] 6D gather memory
- [ ] Hard-fail on missing weights

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/moe_deltakv_paged
nano papers/moe_deltakv_paged/variables.json
cd papers/moe_deltakv_paged
python apply_vars.py --module moe_deltakv_paged
grep -rn "{{" .
python build.py --module moe_deltakv_paged --pdf
```
```

---

## 🎯 خلاصه نهایی

### نکات حیاتی برای مقاله

1. **Simulation است، نه compression** — در abstract و intro صادق باش
2. **Delta-KV memory کاهش نمی‌دهد** — K_hat همان شکل K
3. **4-bit فقط STE است** — در inference FP16/FP32
4. **6D gather memory heavy** — در limitations ذکر کن
5. **`raise` در inject** — fix شود

### نوآوری‌های قابل برجسته‌سازی

1. **MoBA + Delta-KV + 4-bit** — ترکیب سه تکنیک
2. **CLS preservation via router boost** — تضمین صریح
3. **Load balancing برای page routing** — MoE-style loss
4. **STE برای temporal delta** — کاربرد جدید

### تفاوت با مقالات قبلی

| مقاله | Complexity | Real compression? |
|---|---|---|
| A3/AASVD | $\mathcal{O}(L^2)$ | ✓ (weights) |
| TCA | $\mathcal{O}(Lk)$ | ✓ (tokens) |
| Mamba SSM | $\mathcal{O}(L^2 d)$ | ✗ |
| **MoE-DeltaKV** | **$\mathcal{O}(N_p \cdot k \cdot P^2)$** | **✗ (simulation)** |

### هشدار مهم

**این ماژول به عنوان "compression" نیست** — یک **training-time regularization** است که:
- مدل را به low-precision reconstruction مقاوم می‌کند
- در آینده می‌تواند برای compression واقعی استفاده شود
- در حال حاضر memory کاهش نمی‌دهد

---

**قدم بعدی: `run_all_4_moe_deltakv_paged.py` را اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**