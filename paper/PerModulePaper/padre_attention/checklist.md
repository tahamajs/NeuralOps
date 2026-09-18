# Pre-Submission Checklist — PADRe

## Critical Honesty Checks
- [ ] **No kernelization stated in abstract**
- [ ] **Runtime unchanged from softmax stated**
- [ ] **Polynomial breaks for |s| > 5 documented**
- [ ] **Order 2 is default, order 4 is optimal**

## Setup
- [ ] Copied `_TEMPLATE/` to `padre/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module padre`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Polynomial formula stated
- [ ] Kernelization caveat explicit
- [ ] Multi-architecture mentioned

## Method
- [ ] Polynomial kernel defined
- [ ] Max-centering explained
- [ ] ReLU clamp explained
- [ ] All-masked guard
- [ ] Complexity honest

## Results
- [ ] Main results
- [ ] Order ablation
- [ ] Temperature ablation
- [ ] Centering ablation
- [ ] Injection success
- [ ] Equivalence check

## Limitations
- [ ] No kernelization
- [ ] Polynomial breaks
- [ ] Raised on missing
- [ ] RoPE bug

## Bugs to Fix Before Publishing
- [ ] `apply_rotary_pos_emb` fragile `'q_rot' in locals()` check
- [ ] `copy_weights` `raise RuntimeError` → `logger.warning`

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/padre
nano papers/padre/variables.json
cd papers/padre
python apply_vars.py --module padre
grep -rn "{{" .
python build.py --module padre --pdf
```
```

---

## 🎯 خلاصه نهایی

### نکات حیاتی برای مقاله

| هشدار | چرا مهم است |
|---|---|
| **No kernelization** | Runtime همان O(N²) |
| **Polynomial breaks for \|s\| > 5** | محدودیت applicability |
| **Max-centering essential** | برای stability در FP16 |
| **ReLU clamp essential** | برای جلوگیری از NaN |
| **All-masked guard essential** | برای query بدون key |
| **Fragile RoPE check** | باید fix شود |
| **raise در copy_weights** | باید fix شود |

### نوآوری‌های قابل برجسته‌سازی

1. **Polynomial substitution for pretrained models** — اولین کار در این زمینه
2. **Numerical stability tricks** (centering + ReLU + guard)
3. **Output equivalence check** در injection
4. **Multi-architecture weight transfer**

### تفاوت با مقالات قبلی

| مقاله | Kernel | Runtime |
|---|---|---|
| NEXUS | sparse | Sub-quadratic |
| Mamba SSM | SSM-inspired | $\mathcal{O}(L^2 d)$ |
| **PADRe** | **polynomial** | **$\mathcal{O}(N^2 d_h)$** |

### هشدار کلیدی برای نوشتن

**این ماژول speedup نمی‌دهد.** تمام claims "faster inference" باید در paper حذف شوند، مگر اینکه kernelization واقعاً پیاده شود. در abstract و intro **صادقانه** بگو که runtime هنوز O(N²) است.

---

**قدم بعدی: `run_all_4_padre.py` را اجرا کن، اعداد را بردار، در `variables.json` بگذار. اگر می‌خواهی fix bugs (RoPE check + copy_weights raise) را انجام دهم، بگو.**