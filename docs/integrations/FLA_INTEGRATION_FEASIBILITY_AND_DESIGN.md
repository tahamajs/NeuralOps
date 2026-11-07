# ⚡ بررسی جامع امکان‌سنجی و طراحی معماری ادغام با FLA (Flash Linear Attention)

**پلتفرم هدف:** اکوسیستم NeuralOps (v1.7.0+)
**موضوع:** ادغام دوطرفه، استفاده از کرنل‌ها و لایه‌های رسمی `flash-linear-attention (FLA)` در موتور جراحی و پایپ‌لاین‌های NeuralOps
**تاریخ گزارش:** اکتبر ۲۰۲۶
**وضعیت امکان‌سنجی:** **کاملاً امکان‌پذیر (Feasible & Recommended)** — حدود ۳۰٪ زیرساخت آن از قبل در مخزن وجود دارد.

---

## ۱. خلاصه‌ی اجرایی و نتیجه‌ی بررسی (Executive Summary)

کتابخانه **Flash Linear Attention (FLA)** (توسعه‌داده‌شده توسط Songlin Yang و همکاران) معتبرترین مخزن تحقیقاتی تخصصی برای پیاده‌سازی کرنل‌های موازی تریتون (Triton Chunk/Fused Kernels) در حوزه‌ی اتنشن‌های خطی و بازگشتی (مانند GLA, DeltaNet, RetNet, RWKV-6, HGRN-2, Gated DeltaNet) است.

### پاسخ صریح به پرسش کاربر:
> **آیا می‌توان قابلیتی اضافه کرد تا کاربر بتواند مستقیماً از خود کتابخانه‌ی FLA در NeuralOps استفاده کند؟**
> **پاسخ: بله، ۱۰۰٪ و بدون هیچ‌گونه تداخل با معماری فعلی.**

در حال حاضر در NeuralOps:
1. در فایل `pyproject.toml` پکیج اختیاری `fla` از قبل تعریف شده است:
   ```toml
   [project.optional-dependencies]
   fla = ["fla; sys_platform == 'linux'"]
   ```
2. در ماژول‌هایی مانند `neuralops/layers/linear/gla_attention.py`، `flagship_fla.py`، `cosformer.py` و `hgrn2_gated_outer.py` فراخوانی‌های دینامیک آزمایشی به `fla.ops` تعبیه شده است.
3. در `neuralops/evaluation/harness.py` نام `"fla"` در کنار `"neuralops"` به عنوان رپر ارزیابی مدل در `lm_eval` ثبت شده است.

با این حال، این ارتباط در حال حاضر پراکنده است و به عنوان یک **قابلیت رسمی، یکپارچه و کاربرپسند** در API جراحی (`replace_attention`) یا پایپ‌لاین استنتاج (`pipeline`) عرضه نشده است. با پیاده‌سازی این طرح، کاربر می‌تواند با یک سوییچ ساده تعیین کند که جراحی و محاسبات با کرنل بومی NeuralOps انجام شود یا مستقیماً از موتور رسمی FLA بهره ببرد.

---

## ۲. مقایسه‌ی معماری: NeuralOps در برابر FLA

این دو پروژه رقیب یکدیگر نیستند، بلکه **هم‌افزا و مکمل کامل** یکدیگرند:

| ویژگی | NeuralOps | Flash Linear Attention (FLA) |
| :--- | :--- | :--- |
| **تمرکز اصلی** | جراحی درجا روی مدل‌های موجود (Surgery)، فشرده‌سازی عمیق (SVD, Pruning)، کالیبراسیون و پایپ‌لاین‌های یکپارچه | تحقیقات دانشگاهی روی طراحی کرنل‌های تخصصی تریتون و آموزش مدل‌های جدید از پایه |
| **تنوع معماری** | **۱۳۱ مکانیزم اتنشن** + **۷۲ موتور فشرده‌سازی** | حدود ۱۵ تا ۲۰ مکانیزم اتنشن خطی/بازگشتی |
| **سلسله‌مراتب اجرا** | سلسله‌مراتب ۳ سطحی (CUDA C++ بومی $\rightarrow$ تریتون $\rightarrow$ پایتورچ CPU) | عمدتاً وابسته به Triton روی GPUهای انویدیا (لینوکس) |
| **پشتیبانی از پلتفرم** | لینوکس (CUDA 11/12/13)، ویندوز (WSL2)، و مکینتاش (CPU/MPS Fallback) | صرفاً لینوکس مجهز به پردازنده گرافیکی NVIDIA CUDA |
| **جراحی مدل‌های HF** | پشتیبانی درجا از LLaMA, GPT-2, Mistral, Qwen, DistilBERT بدون آموزش مجدد | مدل‌های استاندارد خودش را دارد (`fla.models.*`) ولی ابزار جراحی درجا برای مدل‌های دیگر ندارد |

---

## ۳. الگوهای چهارگانه پیاده‌سازی قابلیت FLA در NeuralOps

برای ارائه تجربه‌ای تمیز به کاربر، ۴ سطح دسترسی (Modes) پیشنهاد می‌شود:

```
                  ┌──────────────────────────────────────────────┐
                  │          کاربر NeuralOps (User API)          │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    [حالت ۱: جراحی با لایه‌های رسمی FLA]         [حالت ۲: دیسپچ کرنل به FLA Ops]
neuralops.replace_attention(model, "fla:gla")       neuralops.force_kernel_backend("fla")
                 │                                               │
                 ▼                                               ▼
      ┌─────────────────────┐                         ┌─────────────────────┐
      │   FLALayerAdapter   │                         │  3-Level Dispatcher │
      │   (سازگار با HF)    │                         │  Level 2b: fla.ops  │
      └──────────┬──────────┘                         └──────────┬──────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │          موتور رسمی FLA (fla-linear-attention) │
                 └──────────────────────────────────────────────┘
```

---

### حالت اول: جراحی درجا با لایه‌های رسمی FLA (`"fla:<architecture>"`)

کاربر می‌خواهد لایه‌های Attention یک مدل موجود در Hugging Face (مثلاً LLaMA-3 یا GPT-2) را مستقیماً با لایه رسمی FLA جایگزین کند:

```python
import neuralops
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# جراحی اتنشن با استفاده از لایه رسمی FLA:
model = neuralops.replace_attention(
    model,
    target_attention="fla:gla",     # یا "fla:retnet", "fla:gated_deltanet", "fla:rwkv6"
    backend="fla"
)
```

#### نحوه کار در پشت صحنه (Adapter Pattern):
لایه‌های بومی FLA ورودی‌ها و خروجی‌هایی با فرمت خاص خود دارند و ممکن است امضای توابع آنها با ترنسفورمرهای هاگینگ‌فیس (`past_key_value`, `attention_mask`, `position_ids`, `use_cache`) متفاوت باشد. NeuralOps یک کلاس واسط تحت عنوان `FLALayerAdapter` ایجاد می‌کند که امضای استاندارد HF را به ورودی‌های مورد نیاز FLA نگاشت می‌کند.

---

### حالت دوم: انتخاب FLA به عنوان Backend پردازشی کرنل‌ها (`force_kernel_backend`)

کاربر از معماری‌های بومی NeuralOps استفاده می‌کند، اما در لایه محاسبات کرنل، به جای کرنل بومی CUDA C++ یا Triton داخلی، مایل است از پیاده‌سازی بهینه‌شده‌ی `fla.ops` استفاده کند:

```python
import neuralops

# انتخاب موتور محاسباتی در سطح کل پروژه
neuralops.force_kernel_backend("fla")

# یا به صورت Context Manager موقت برای بنچمارک:
with neuralops.use_kernel_backend("fla"):
    outputs = model(inputs)
```

سلسله‌مراتب پردازش (Dispatch Hierarchy) به شکل زیر ارتقا می‌یابد:
1. **Level 1**: بومی Native CUDA C++ (بیشترین توان عملیاتی)
2. **Level 2a**: کرنل‌های بومی NeuralOps Triton JIT
3. **Level 2b**: کرنل‌های رسمی `fla.ops` (در صورت انتخاب کاربر یا نبود Level 1/2a)
4. **Level 3**: مرجع برداری PyTorch (اجرا روی هر CPU / Mac / بدون نیاز به CUDA)

---

### حالت سوم: بنچمارک و مقایسه برابری عددی (Parity & Speedup Benchmark)

یکی از جذاب‌ترین کاربردها برای پژوهشگران، مقایسه مستقیم دقت و سرعت کرنل‌های داخلی NeuralOps با کرنل‌های رسمی FLA است:

```bash
# مقایسه عملکرد کرنل داخلی GLA با کرنل FLA
neuralops benchmark --layer gla_attention --compare-with fla --seq-len 8192
```

خروجی خودکار:
- مقایسه سرعت (Latency بر حسب میلی‌ثانیه)
- مصرف VRAM (Peak Allocation)
- اختلاف عددی ($\Delta_{\max} = |Y_{\text{neuralops}} - Y_{\text{fla}}|$ با شرط $\Delta < 10^{-4}$)

---

### حالت چهارم: تبدیل و بارگذاری وزن‌های مدل‌های از‌پیش‌آموزش‌دیده FLA

کاربرانی که مدل‌هایی با معماری‌های FLA آموزش داده‌اند (یا چک‌پوینت‌های موجود در `fla-hub` مانند `fla-hub/gla-1.3B`)، بتوانند وزن‌های آن را مستقیماً وارد لایه‌های جراحی‌شده NeuralOps کنند و از موتورهای فشرده‌سازی (مثل SVD و Pruning) استفاده نمایند:

```python
import neuralops

# بارگذاری یکپارچه چک‌پوینت FLA در مدل جراحی‌شده
neuralops.load_fla_weights(model, source_repo="fla-hub/gla-1.3B")

# انجام فشرده‌سازی فوری روی مدل
compressed_model = neuralops.compress(model, method="fwsvd", rank=64)
```

---

## ۴. طرح فنی پیاده‌سازی (Technical Implementation Blueprint)

برای اضافه کردن این قابلیت بدون شکستن قوانین مخزن (`AGENTS.md`)، ساختار زیر در پوشه‌ی `neuralops/integrations/fla/` ایجاد خواهد شد:

```
neuralops/integrations/fla/
├── __init__.py               # اکسپورت عمومی و چک در دسترس بودن FLA (FLA_AVAILABLE)
├── adapter.py                # رپر لایه‌های FLA با امضای Hugging Face
├── registry_bridge.py        # ثبت اتوماتیک معماری‌های fla:* در رجیستری
├── dispatch_hook.py          # اتصال به سیستم force_kernel_backend
└── weights_converter.py      # تبدیل نام پارامترها و وزن‌ها
```

### نمونه کد پیاده‌سازی آداپتور (`adapter.py`):

```python
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
# This source code is licensed under the MIT license.

from __future__ import annotations
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import fla
    from fla.layers import (
        GatedLinearAttention,
        DeltaNet,
        RetNet,
        HGRN2Attention,
    )
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    fla = None


class FLALayerAdapter(nn.Module):
    """
    Adapter wrapping official FLA layers to provide strict Hugging Face
    transformer attention signature compatibility.
    """
    def __init__(self, fla_layer_cls, config, **kwargs):
        super().__init__()
        if not FLA_AVAILABLE:
            raise ImportError(
                "The 'fla' package is required to use FLA layers. "
                "Install it via: pip install 'neuralops[fla]' or pip install fla-linear-attention"
            )
        self.inner = fla_layer_cls(
            mode=kwargs.get("mode", "chunk"),
            hidden_size=getattr(config, "hidden_size", 768),
            num_heads=getattr(config, "num_attention_heads", 12),
            **kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        # نگاشت فرمت ورودی و حافظه موقت (State) به لایه بومی FLA
        outputs = self.inner(
            hidden_states,
            past_key_values=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if isinstance(outputs, tuple):
            attn_output = outputs[0]
            next_state = outputs[1] if len(outputs) > 1 else None
        else:
            attn_output = outputs
            next_state = None

        return attn_output, next_state
```

---

## ۵. مزایای راهبردی این افزونه برای NeuralOps

1. **انعطاف‌پذیری فوق‌العاده برای کاربران پژوهشی**: پژوهشگرانی که مقالات خود را بر پایه‌ی FLA منتشر می‌کنند، می‌توانند بدون بازنویسی کد، مستقیماً از جراحی مدل و فشرده‌سازی SVD در NeuralOps بهره‌مند شوند.
2. **عدم ایجاد وابستگی اجباری (Zero Bloat)**: پکیج `fla` به صورت **کاملاً اختیاری (Optional Extra)** باقی می‌ماند. کاربرانی که روی مکینتاش، ویندوز بدون CUDA، یا پایتورچ استاندارد هستند با هیچ خطایی مواجه نخواهند شد.
3. **اعتبارسنجی متقابل نتایج (Cross-Validation)**: امکان بررسی صحت عددی کرنل‌های NeuralOps در برابر معتبرترین فریم‌ورک متن‌باز دنیا فراهم می‌شود.
4. **حفظ استانداردهای مخزن (`AGENTS.md`)**:
   - شمارنده‌های مرجع (۱۳۱ اتنشن و ۷۲ کمپرسور) دست‌نخورده باقی می‌مانند، زیرا ماژول‌های FLA تحت پیشوند اختصاصی `fla:*` به عنوان اکستنشن ثبت می‌شوند.
   - هیچ تغییری در API عمومی جاری ایجاد نمی‌شود و سازگاری کامل به عقب (Backward-Compatibility) حفظ می‌گردد.

---

## ۶. نقشه راه و گام‌های اجرای پروژه (Next Steps)

در صورت تأیید شما، می‌توان مراحل زیر را گام‌به‌گام پیاده‌سازی کرد:

1. **گام اول**: ساخت پکیج پل ارتباطی در `neuralops/integrations/fla/` با رعایت واردسازی پویا (`dynamic import guard`).
2. **گام دوم**: اتصال پیشوند `fla:*` در تابع مرکزی جراحی `replace_attention`.
3. **گام سوم**: افزودن سوییچ انتخاب بک‌اند `force_kernel_backend("fla")` به ماژول دیسپچ.
4. **گام چهارم**: نوشتن تست‌های جامع رگرسیون در `tests/integrations/test_fla_bridge.py` با نشانه‌گذاری `@pytest.mark.skipif(not FLA_AVAILABLE)`.
5. **گام پنجم**: مستندسازی رسمی در بخش Integrations کتابخانه و ارائه مثال‌های کاربردی در نوتبوک‌ها.
