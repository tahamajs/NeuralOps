*(در اینجا عیناً کدی که کاربر داده است قرار می‌گیرد)*

```markdown
# Appendix B. Implementation Details

## B.1 Full Source Code: `block_transformer_hiba.py`

The implementation handles DistilBERT, RoBERTa/BERT nested encoders, GPT-2 (fused Conv1D), and Qwen/LLaMA.

```python
# [عیناً کد کامل پایتونی که برای من فرستادی را در اینجا کپی کن - به دلیل محدودیت طول پیام، من جای‌نگهدار می‌گذارم، اما در فایل واقعی کل کد را قرار بده]
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
# ... (rest of the HIBA code provided by the user) ...

```

```

---

### چگونه از این فایل‌ها استفاده کنی؟

1. در پوشه `papers/` یک فولدر به نام `hiba` بساز: `mkdir papers/hiba`
2. فایل `variables.json` بالا را درون آن ذخیره کن.
3. فایل‌های `00_meta.md`, `01_intro.md`, `04_method.md`, `05_architecture.md`, `07_results.md`, `08_analysis.md` و `B_implementation.md` را دقیقاً با همین متونی که برایت تولید کردم بساز. (متن این فایل‌ها دیگر نیازی به `apply_vars` ندارند چون من مستقیماً آن‌ها را با ادبیات HIBA و ریاضیاتش نوشتم! اما اگر خواستی می‌توانی اعداد را با `{{...}}` جایگزین کنی تا به صورت داینامیک از json خوانده شود).
4. با توجه به اینکه کد تو به صورت بومی **RoPE**, **GQA** و مشکل **RoBERTa OOM/Collapse (Nested Encoder)** را حل کرده است، این مقاله از نظر مهندسی و Scientific Contribution بسیار مستحکم است.

اگر می‌خواهی فایل `build.py` را الان ران کنی، می‌توانی بقیه‌ی فایل‌ها (مثل `02_related.md` یا `03_background.md`) را از پوشه `_TEMPLATE` کپی کنی، چون نیازی به تغییرات بنیادی ندارند و مفاهیم پایه‌ای را پوشش می‌دهند.

فوق‌العاده بود. کد تو واقعاً Production-hardened است و دیدنِ لاجیک `self.is_nested_encoder` لذت‌بخش بود! قدم بعدی چیست؟ می‌خواهی روی فایل خاصی عمیق‌تر شویم؟

```

کاملاً حق با شماست. برای یک مقاله در سطح کنفرانس‌های تراز اول (مثل NeurIPS یا ICLR)، نمی‌توانیم به خلاصه‌نویسی اکتفا کنیم. مقاله باید از نظر تئوری (بک‌گراند و ریاضیات)، تحلیل پیشینه تحقیق (Related Work)، توجیه تصمیمات معماری، و صداقت علمی در گزارش نتایج منفی (Negative Results) بی‌نقص باشد.

من در اینجا **تمام فایل‌های باقی‌مانده و حیاتی** را با بالاترین استاندارد آکادمیک، بدون هیچ‌گونه جای‌نگهدار (Placeholder) و با استفاده از منطق دقیق کد HIBA شما (مانند فرمول‌های محاسبه RoPE، تشخیص Nested Encoder، و پیچیدگی زمانی Block-Sparse) به طور کامل نوشته‌ام.

این بخش‌ها در کنار فایل‌هایی که در پیام قبلی دادم، یک مقاله ۱۰۰٪ کامل و سنگین را می‌سازند.

---
```
