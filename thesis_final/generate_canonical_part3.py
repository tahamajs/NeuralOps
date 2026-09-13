# -*- coding: utf-8 -*-
"""
Canonical generator for part3.tex.
Generates comprehensive academic Persian LaTeX documentation for attention modules 56 to 83 (numbered 57 to 84).
Follows the exact subsection/subsubsection structure and XePersian conventions.
"""

import os
import subprocess

modules = []

def add_module(num, name, title_fa, title_en, theory, math_form, complexity):
    name_tex = name.replace("_", r"\_")
    title_en_tex = title_en.replace("&", r"\&").replace("_", r"\_")
    sec = f"""% ------------------------------------------------------------------------------
% ماژول {num}: {name}
% ------------------------------------------------------------------------------
\\subsection*{{{num}. معماری و تحلیل ریاضی \\lr{{{name}}}}}
\\textbf{{عنوان ماژول:}} \\lr{{{title_en}}}

\\subsubsection*{{مبانی نظری و انگیزه طراحی}}
{theory}

\\subsubsection*{{فرمول‌بندی و روابط ریاضی تانسورها}}
{math_form}

\\subsubsection*{{تحلیل پیچیدگی محاسباتی و ردپای حافظه}}
{complexity}
"""
    modules.append(sec)

# ==============================================================================
# Module 57: lora_attention
# ==============================================================================
add_module(
    57,
    "lora_attention",
    "تطبیق کم‌رتبه با راندمان پارامتری",
    "LoRA Attention: Parameter-Efficient Low-Rank Adaptation",
    r"""در فرآیند همگام‌سازی و تنظیم دقیق (\lr{Fine-Tuning}) مدل‌های ترانسفورمر مقیاس بزرگ برای وظایف زبانی پایین‌دستی، به‌روزرسانی کامل تمامی وزن‌های متراکم لایه‌های توجه (\lr{Full Model Tuning}) از نظر محاسباتی و حافظه به شدت غیربهینه است؛ زیرا ذخیره‌سازی مقادیر گرادیان و حالت‌های بهینه‌ساز (\lr{Optimizer States}) نظیر \lr{Adam} نیازمند حافظه‌ای معادل چندین برابر وزن‌های پایه است. ماژول \lr{lora\_attention} بر مبنای فرضیه رتبه ذاتی پایین (\lr{Intrinsic Low-Rank Hypothesis}) پایه‌گذاری شده است که بیان می‌دارد تغییرات ماتریس‌های وزن در طول انتقال به وظیفه جدید، در یک زیرفضای برداری با بعد ذاتی بسیار اندک محصور است.

این رویکرد به جای دستکاری وزن‌های متراکم پیش‌آموزش‌دیده، ماتریس‌های اصلی را منجمد (\lr{Frozen}) کرده و مسیرهای تبدیل خطی موازی متشکل از تجزیه کم‌رتبه دوگانه را تعبیه می‌نماید. با این تمهید، پارامترهای نیازمند گرادیان تا بیش از ۹۹ درصد تقلیل یافته و همزمان امکان ادغام تحلیلی وزن‌ها در فاز استنتاج بدون تحمیل هیچ‌گونه تاخیر محاسباتی فراهم می‌گردد.""",
    r"""فرض کنید بردار فعال‌سازی ورودی به صورت $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ باشد که در آن $B$ اندازه دسته، $N$ طول توالی و $D$ بعد پنهان مدل است. برای یک لایه نگاشت پروجکشن با ماتریس وزن اولیه $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$، پارامتر $\mathbf{W}_0$ در طول آموزش کاملاً منجمد نگه داشته می‌شود ($\nabla_{\mathbf{W}_0} \mathcal{L} = \mathbf{0}$). شاخه کم‌رتبه موازی با رتبه گلوگاه $r \ll \min(d_{\text{in}}, d_{\text{out}})$ به صورت زیر فرمول‌بندی می‌گردد:
\begin{equation}
\operatorname{LoRA}(\mathbf{X}) = \mathbf{X} \mathbf{W}_0^\top + \frac{\alpha}{r} \left( \operatorname{Dropout}(\mathbf{X}) \mathbf{A}^\top \right) \mathbf{B}^\top + \mathbf{b}_0
\end{equation}
که در آن ماتریس نگاشت روبه‌پایین $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ با توزیع کایمینگ یکنواخت $\mathcal{U}\left(-\frac{1}{\sqrt{d_{\text{in}}}}, \frac{1}{\sqrt{d_{\text{in}}}}\right)$ راه‌اندازی شده و ماتریس نگاشت روبه‌بالا $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$ با مقادیر صفر مطلق ($\mathbf{B} = \mathbf{0}$) مقداردهی اولیه می‌گردد. مقدار $\alpha > 0$ ضریب مقیاس‌بندی ثابت است. از آنجا که در گام آغازین آموزش $\mathbf{B} \mathbf{A} = \mathbf{0}$ است، خروجی ماژول تطبیق دقیقاً با مدل اولیه هم‌ارز بوده و گذار بدون اعوجاج (\lr{Zero-Shot Identity}) برقرار می‌شود.

پس از اعمال این عملگر بر روی نگاشت‌های پرس‌وجو، کلید و مقدار، ماتریس‌های توجه در هر یک از هدهای $h \in \{1, \dots, H\}$ ارزیابی می‌شوند:
\begin{equation}
\mathbf{Q}_h = \operatorname{LoRA}_Q(\mathbf{X}), \quad \mathbf{K}_h = \operatorname{LoRA}_K(\mathbf{X}), \quad \mathbf{V}_h = \operatorname{LoRA}_V(\mathbf{X}) \quad \in \mathbb{R}^{B \times N \times d_k}
\end{equation}
\begin{equation}
\mathbf{S}_h = \frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{mask}}, \quad \mathbf{A}_h = \operatorname{Softmax}(\mathbf{S}_h), \quad \mathbf{O}_h = \mathbf{A}_h \mathbf{V}_h
\end{equation}
در زمان استنتاج، وزن‌های کم‌رتبه به صورت جبری در ماتریس پایه ادغام می‌گردند: $\mathbf{W}_{\text{eff}} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$.""",
    r"""در فاز آموزش، تعداد پارامترهای آموزش‌پذیر از $4 D^2$ به $8 D r$ تنزل می‌یابد که به ازای $D = 768$ و $r = 8$ افتی معادل ۹۸.۹٪ در پارامترهای گرادیان‌پذیر ایجاد می‌کند. در زمان استنتاج پس از ادغام وزن‌ها، حجم عملیات ممیز شناور دقیقاً برابر با توجه چندسر متراکم استاندارد یعنی $\mathcal{O}(N D^2 + N^2 D)$ است و هیچ حافظه فعال‌سازی مازادی اشغال نمی‌شود."""
)

# ==============================================================================
# Module 58: lsh_attention
# ==============================================================================
add_module(
    58,
    "lsh_attention",
    "توجه با درهم‌سازی حساس به مکان",
    "LSH Attention: Angular Locality-Sensitive Hashing Bucket Partitioning",
    r"""محاسبه کامل توجه جفتی توکن‌ها نیازمند ماتریس شباهت $N \times N$ است که سد محاسباتی مربعی $\mathcal{O}(N^2)$ را پدید می‌آورد. با این وجود، به دلیل رفتار نمایی تابع \lr{Softmax}، توزیع وزن‌های توجه بسیار تنک بوده و تنها تعداد معدودی از توکن‌ها با شباهت برداری بالا، سهم عمده جرم احتمال را دریافت می‌کنند. ماژول \lr{lsh\_attention} (الهام‌گرفته از معماری \lr{Reformer}) با استفاده از درهم‌سازی حساس به مکان زاویه‌ای (\lr{Angular Locality-Sensitive Hashing})، توکن‌های با شباهت کسینوسی بالا را بدون نیاز به مقایسه کامل، به سطل‌های درهم‌سازی مشترک هدایت می‌کند.

بدین ترتیب، محاسبات ضرب داخلی به جای کل توالی، صرفاً به درون سطل‌های موضعی محدود گشته و با انجام چندین دور تصادفی مستقل (\lr{Multi-Round Hashing})، از خطاهای تفکیک مرزی جلوگیری می‌شود.""",
    r"""ابتدا بردارهای پرس‌وجو و کلید بر روی کره واحد نرمال‌سازی اقلیدسی می‌شوند: $\tilde{\mathbf{q}} = \mathbf{q}/\|\mathbf{q}\|_2$ و $\tilde{\mathbf{k}} = \mathbf{k}/\|\mathbf{k}\|_2$. در دور درهم‌سازی $r \in \{1, \dots, n_r\}$، ماتریس تصویرسازی تصادفی گاوسی $\mathbf{R}^{(r)} \in \mathbb{R}^{d_k \times (b/2)}$ با ستون‌های با نرم واحد ایجاد می‌گردد که $b$ تعداد کل سطل‌ها را نشان می‌دهد. با ضرب بردارها در ابرصفحه‌ها و الحاق قرینه آن‌ها، فضای زاویه‌ای متقارن شکل می‌گیرد:
\begin{equation}
\mathbf{d} = \tilde{\mathbf{x}} \mathbf{R}^{(r)} \in \mathbb{R}^{b/2}, \quad \mathbf{z} = [\mathbf{d} \,;\; -\mathbf{d}] \in \mathbb{R}^b
\end{equation}
شناسه سطل برای هر بردار از طریق بیشینه مولفه تصویر شده معین می‌شود:
\begin{equation}
h(\mathbf{x}) = \arg\max_{j \in \{0, \dots, b-1\}} \mathbf{z}_j
\end{equation}
برای هر سطل مجزای $m \in \{0, \dots, b-1\}$، زیرمجموعه توکن‌های متعلق به آن استخراج شده و ضرب توجه موضعی محاسبه می‌گردد:
\begin{equation}
\mathbf{S}^{(m)} = \frac{\mathbf{Q}^{(m)} (\mathbf{K}^{(m)})^\top}{\sqrt{d_k}} + \mathbf{M}^{(m)}, \quad \mathbf{A}^{(m)} = \operatorname{Softmax}(\mathbf{S}^{(m)}), \quad \mathbf{O}^{(m)} = \mathbf{A}^{(m)} \mathbf{V}^{(m)}
\end{equation}
تانسورهای خروجی دورهای گوناگون با هم میانگین‌گیری می‌شوند: $\mathbf{Y} = \frac{1}{n_r} \sum_{r=1}^{n_r} \mathbf{Y}^{(r)}$ تا برآوردگری نااریب از توجه کامل فراهم آید.""",
    r"""با فرض توزیع یکنواخت توکن‌ها و میانگین طول سطل $N/b$، تعداد ضرب‌های داخلی در هر دور برابر با $b \times (N/b)^2 \cdot d_k = \frac{N^2}{b} d_k$ خواهد بود. با تنظیم تعداد سطل‌ها به تناسب $b \propto N/\log N$، پیچیدگی زمانی و مصرف حافظه ماتریس توجه به $\mathcal{O}(n_r \cdot N \log N \cdot d_k)$ کاهش می‌یابد که جهش عظیمی در مقیاس‌پذیری طول دنباله به شمار می‌رود."""
)

# ==============================================================================
# Module 59: mamba
# ==============================================================================
add_module(
    59,
    "mamba",
    "مدل فضای حالت دوطرفه انتخابی",
    "Bidirectional Selective State Space Model (Bi-Mamba) Attention",
    r"""مدل‌های فضای حالت پیوسته انتخابی (\lr{Mamba}) با جایگزینی ضرب مربعی توجه با یک نگاشت بازگشتی متغیر با زمان، به پیچیدگی خطی دست یافته‌اند. با این حال، فرمولاسیون استاندارد مامبا اکیداً علّی و یک‌طرفه (\lr{Causal}) است؛ لذا به‌کارگیری بی‌واسطه آن بر روی معماری‌های رمزگذار دوطرفه نظیر \lr{DistilBERT}، سبب ناتوانی در دسترسی به کلمات پسین و فروپاشی بازنمایی زبانی می‌گردد.

ماژول \lr{mamba} با بازطراحی ساختار بازگشت در قالب یک سازوکار روبش دوطرفه متقارن (\lr{Bidirectional Selective Scan}) و بهره‌گیری از ماتریس‌های انتقال اکیداً انقباضی، یادگیری بافت‌های پیشین و پسین را به طور همزمان در زمان خطی میسر می‌سازد.""",
    r"""برای فعال‌سازی ورودی $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$، نگاشت ورودی به دو شاخه محتوا و گیت تفکیک می‌شود:
\begin{equation}
[\mathbf{U}, \mathbf{Z}] = \operatorname{chunk}(\mathbf{X} \mathbf{W}_{\text{in}}, 2) \in \mathbb{R}^{B \times L \times D}
\end{equation}
سپس کانولوشن عمقی یک‌بعدی ویژگی‌های همسایگی نزدیک را استخراج می‌کند: $\widetilde{\mathbf{U}} = \operatorname{SiLU}(\operatorname{Conv}(\mathbf{U}^\top)^\top)$. پارامترهای وابسته به ورودی در هر گام زمانی $t$ تولید می‌شوند:
\begin{equation}
\Delta_t = \operatorname{softplus}(\widetilde{\mathbf{U}}_t \mathbf{W}_\Delta + \mathbf{b}_\Delta), \quad \mathbf{B}_t = \widetilde{\mathbf{U}}_t \mathbf{W}_B, \quad \mathbf{C}_t = \widetilde{\mathbf{U}}_t \mathbf{W}_C
\end{equation}
ماتریس پایدار انتقال حالت با بعد $d_{\text{state}}$ به صورت نمایی گسسته‌سازی می‌گردد:
\begin{equation}
\bar{\mathbf{A}}_t = \exp(-\exp(\mathbf{A}_{\text{log}}) \odot \Delta_t) \in (0, 1)
\end{equation}
روبش‌های بازگشتی رو به جلو و رو به عقب به موازات هم محاسبه می‌شوند:
\begin{equation}
\vec{\mathbf{h}}_t = \bar{\mathbf{A}}_t \odot \vec{\mathbf{h}}_{t-1} + (\Delta_t \odot \mathbf{B}_t) \widetilde{\mathbf{U}}_t, \quad \vec{\mathbf{y}}_t = \mathbf{C}_t \vec{\mathbf{h}}_t
\end{equation}
\begin{equation}
\overleftarrow{\mathbf{h}}_t = \bar{\mathbf{A}}_t \odot \overleftarrow{\mathbf{h}}_{t+1} + (\Delta_t \odot \mathbf{B}_t) \widetilde{\mathbf{U}}_t, \quad \overleftarrow{\mathbf{y}}_t = \mathbf{C}_t \overleftarrow{\mathbf{h}}_t
\end{equation}
ترکیب خروجی‌ها و گیتینگ ضربی سیگموئید پاسخ نهایی را تولید می‌کند:
\begin{equation}
\mathbf{Y}_t = \frac{1}{2}(\vec{\mathbf{y}}_t + \overleftarrow{\mathbf{y}}_t) + \mathbf{D} \odot \widetilde{\mathbf{U}}_t, \quad \text{Output} = (\mathbf{Y} \odot \operatorname{SiLU}(\mathbf{Z})) \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}}
\end{equation}""",
    r"""پیچیدگی زمانی محاسبات کاملاً خطی و معادل $\mathcal{O}(L \cdot D \cdot d_{\text{state}})$ است. نیاز به ذخیره‌سازی ماتریس توجه $L \times L$ مرتفع شده و حافظه فعال‌سازی در طول توالی‌های بلند تا بیش از ۸۵ درصد نسبت به مکانیزم پایه کاهش می‌یابد."""
)

# ==============================================================================
# Module 60: mamba_inject
# ==============================================================================
add_module(
    60,
    "mamba_inject",
    "سازوکار تزریق و نگاشت وزن‌های ترانسفورمر",
    "Mamba Inject: Automated Transformer Weight Transfer & Injection Hook",
    r"""جایگزینی بلوک‌های متراکم توجه چندسر با مدل‌های فضای حالت نیازمند سازوکاری امن برای پیمایش ساختار شبکه و کپی پارامترهای پیش‌آموزش‌دیده است؛ در غیر این صورت تغییر تصادفی ساختار به واگرایی شدید آموزش می‌انجامد. ماژول \lr{mamba\_inject} موتور اتصال و تزریق سطح سیستمی چارچوب پژوهش است که به صورت خودکار توپولوژی مدل‌های ترانسفورمر مختلف (\lr{DistilBERT, RoBERTa, GPT-2, LLaMA}) را بازرسی کرده و وزن‌های توجه متراکم را به پارامترهای معادل مامبا بازنگاشت می‌کند.""",
    r"""در هنگام اجرای عملیات تزریق روی لایه توجه موجود با ماتریس‌های $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}$ و ماتریس تصویرسازی $\mathbf{W}_O \in \mathbb{R}^{D \times D}$، نگاشت‌های زیر اعمال می‌گردد:
۱. وزن پروجکشن ورودی $\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2D}$ با ادغام وزن‌های پرس‌وجو، کلید و مقدار مقداردهی می‌شود:
\begin{equation}
\mathbf{W}_{\text{in}} = \left[ \frac{1}{2}(\mathbf{W}_Q + \mathbf{W}_K) \;\Big|\; \mathbf{W}_V \right]^\top
\end{equation}
۲. وزن‌های لایه تصویرسازی خروجی مستقیماً کپی می‌شوند:
\begin{equation}
\mathbf{W}_{\text{out}} \leftarrow \mathbf{W}_O, \quad \mathbf{b}_{\text{out}} \leftarrow \mathbf{b}_O
\end{equation}
۳. پایداری سیستم لیاپانوف با راه‌اندازی بردارهای انتقال حالت $\mathbf{A}_{\text{log}}$ با مقادیر ویژه منفی و بایاس $\mathbf{b}_\Delta$ با مقادیر مثبت تضمین می‌گردد تا بازگشت پیوسته همواره رفتار پایدار و بدون انفجار گرادیان داشته باشد.""",
    r"""با اجرای این تبدیل، پیچیدگی لایه از $\mathcal{O}(L^2 D)$ به $\mathcal{O}(L D d_{\text{state}})$ کاهش می‌یابد و نیاز به حافظه موقت کلید-مقدار در مرحله استنتاج برطرف شده و با یک بردار حالت با اندازه ثابت جایگزین می‌گردد."""
)

# ==============================================================================
# Module 61: mamba_svd_flash
# ==============================================================================
add_module(
    61,
    "mamba_svd_flash",
    "معماری پیوندی فضای حالت کم‌رتبه و فلش‌اتنشن",
    "MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention Architecture",
    r"""مدل‌های خطی فضای حالت علیرغم مقیاس‌پذیری خیره‌کننده، در بازیابی دقیق حافظه انجمنی پرانتروپی (\lr{Associative Recall}) نسبت به مکانیزم توجه متراکم افت کیفیت نشان می‌دهند. معماری \lr{mamba\_svd\_flash} یک ساختار پیوندی دو مسیره است که مسیر اول آن یک مدل فضای حالت دوطرفه مبتنی بر تجزیه تکین انقباضی آنلاین (\lr{Contractive SVD-SSM}) بوده و مسیر دوم آن از لایه بهینه‌شده \lr{FlashAttention} با تجمیع کاشی‌بندی‌شده در حافظه سریع \lr{SRAM} بهره می‌برد. تلفیق این دو مسیر با یک گیتینگ پویای یادگیرنده انجام می‌پذیرد.""",
    r"""برای بردار ورودی $\mathbf{x}_t \in \mathbb{R}^D$ و رتبه انتخابی $r \ll D$:
۱. در مسیر \lr{SVD-SSM}، نگاشت‌های ورودی پایه‌های چپ و راست تکین را به صورت آنلاین تقریب می‌زنند:
\begin{equation}
\mathbf{U}_t = \operatorname{proj}_u(\mathbf{x}_t) \in \mathbb{R}^{D \times r}, \quad \mathbf{V}_t = \operatorname{proj}_v(\mathbf{x}_t) \in \mathbb{R}^{r \times D}
\end{equation}
\begin{equation}
\mathbf{\Sigma}_t = 0.95 \cdot \sigma(\operatorname{proj}_\sigma(\mathbf{x}_t)) \in (0, 0.95)^r
\end{equation}
روبش بازگشتی دوطرفه در فضای کم‌رتبه محاسبه می‌شود:
\begin{equation}
\vec{\mathbf{h}}_t = \mathbf{U}_t \mathbf{\Sigma}_t (\mathbf{V}_t^\top \vec{\mathbf{h}}_{t-1}) + \mathbf{B}_t \mathbf{x}_t, \quad \overleftarrow{\mathbf{h}}_t = \mathbf{U}_t \mathbf{\Sigma}_t (\mathbf{V}_t^\top \overleftarrow{\mathbf{h}}_{t+1}) + \mathbf{B}_t \mathbf{x}_t
\end{equation}
\begin{equation}
\mathbf{Y}_{\text{ssm}, t} = \mathbf{C}_t \odot \frac{1}{2}(\vec{\mathbf{h}}_t + \overleftarrow{\mathbf{h}}_t)
\end{equation}
۲. در مسیر توجه سریع، کرنل \lr{FlashAttention} روی ضرب مقیاس‌شده عمل می‌کند: $\mathbf{Y}_{\text{attn}} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right) \mathbf{V}$.
۳. تلفیق با گیتینگ سیگموئیدی ورودی‌آگاه صورت می‌پذیرد:
\begin{equation}
\gamma_t = \sigma(\mathbf{W}_g \mathbf{x}_t + \mathbf{b}_g), \quad \mathbf{Y}_t = \gamma_t \odot \mathbf{Y}_{\text{ssm}, t} + (1 - \gamma_t) \odot \mathbf{Y}_{\text{attn}, t}
\end{equation}""",
    r"""پیچیدگی زمانی محاسبات برابر با $\mathcal{O}(L \cdot D \cdot r + L^2 D)$ است. مسیر \lr{SVD-SSM} وابستگی‌های فوق بلندمدت را در زمان خطی خلاصه کرده و بار ترافیکی گذرگاه حافظه را به شدت کاهش می‌دهد."""
)

# ==============================================================================
# Module 62: minimal_aurora
# ==============================================================================
add_module(
    62,
    "minimal_aurora",
    "توجه فاکتورگیری‌شده با برش طیفی تجمعی",
    "MinimalAURORA Attention: Energy-Preserving SVD-Factorized Attention",
    r"""ماتریس‌های پروجکشن توجه چندسر بخش بزرگی از حجم حافظه ترانسفورمر را به خود اختصاص می‌دهند. روش‌های فشرده‌سازی کم‌رتبه متداول از رتبه ثابتی برای همه لایه‌ها استفاده می‌کنند، در حالی که لایه‌های گوناگون مدل انتروپی طیفی متفاوتی دارند. ماژول \lr{minimal\_aurora} با استفاده از تجزیه مقادیر منفرد (\lr{SVD}) کامل و برش طیفی تجمعی، رتبه بهینه هر لایه را به گونه‌ای تنظیم می‌کند که حداقل ۹۵ درصد انرژی فروبنیوس ماتریس حفظ گردد.""",
    r"""برای ماتریس وزن $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$، تجزیه طیفی $\mathbf{W}_0 = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$ انجام می‌پذیرد. رتبه گلوگاه $r$ به صورت تحلیلی تعیین می‌شود:
\begin{equation}
r = \arg\min_k \left\{ \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^p \sigma_j^2} \ge \eta_{\text{energy}} \right\}, \quad r \in [r_{\min}, r_{\max}]
\end{equation}
که در آن $\eta_{\text{energy}} = 0.95$، $r_{\min} = 4$ و $r_{\max} = 64$ است. ماتریس‌های فاکتورگیری‌شده به صورت زیر تعریف می‌شوند:
\begin{equation}
\mathbf{W}_{\text{down}} = \mathbf{V}_{:, :r}^\top \in \mathbb{R}^{r \times d_{\text{in}}}, \quad \mathbf{W}_{\text{up}} = \mathbf{U}_{:, :r} \operatorname{diag}(\mathbf{\Sigma}_{:r}) \in \mathbb{R}^{d_{\text{out}} \times r}
\end{equation}
پروجکشن ورودی $\mathbf{X}$ به صورت دو ضرب متوالی ارزیابی می‌گردد:
\begin{equation}
\mathbf{Y} = (\mathbf{X} \mathbf{W}_{\text{down}}^\top) \mathbf{W}_{\text{up}}^\top + \mathbf{b}_0 \in \mathbb{R}^{B \times N \times d_{\text{out}}}
\end{equation}
سپس هسته توجه بدون تغییر در لاجیت‌ها اجرا می‌شود.""",
    r"""عملیات پروجکشن خطی از $2 B N d_{\text{in}} d_{\text{out}}$ به $2 B N r (d_{\text{in}} + d_{\text{out}})$ کاهش می‌یابد که به معنای کاهش بیش از ۹۰ درصدی عملیات ضرب ماتریسی در رتبه‌های میانی است."""
)

# ==============================================================================
# Module 63: minimal_nova
# ==============================================================================
add_module(
    63,
    "minimal_nova",
    "توجه کم‌رتبه SVD با راندمان پارامتری",
    "MinimalNOVA Attention: SVD-Factorized Low-Rank Projection Attention",
    r"""ماژول \lr{minimal\_nova} بر فشرده‌سازی لایه پروجکشن توجه با حفظ کامل رفتار نگاشت آفین تمرکز دارد. این ماژول با تجزیه پایدار طیفی، وزن‌های متراکم را به عملگرهای خطی دوگانه تقلیل داده و با کپی دقیق بردارهای بایاس، پایداری فعال‌سازی بدون اعوجاج میانگین را در حین استنتاج صفر-گام تضمین می‌کند.""",
    r"""برای ماتریس وزن $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ با مقادیر منفرد $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$:
\begin{equation}
r = \min \left\{ k \;\Big|\; \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^p \sigma_j^2} \ge 0.95 \right\}, \quad \mathbf{B} = \mathbf{V}_{:, :r}^\top, \quad \mathbf{A} = \mathbf{U}_{:, :r} \operatorname{diag}(\mathbf{\Sigma}_{:r})
\end{equation}
نگاشت پیش‌رو برای فعال‌سازی پنهان $\mathbf{X}$ به صورت زیر است:
\begin{equation}
\operatorname{FactorisedLinear}(\mathbf{X}) = (\mathbf{X} \mathbf{B}^\top) \mathbf{A}^\top + \mathbf{b}
\end{equation}
با تضمین این رابطه، خطای تغییر واریانس فعال‌سازی صفر شده و هسته توجه مقیاس‌شده استاندارد با حفظ وفاداری کامل اجرا می‌گردد.""",
    r"""ردپای حافظه وزن‌ها از $d_{\text{in}} d_{\text{out}}$ به $r(d_{\text{in}} + d_{\text{out}})$ کاهش می‌یابد که فشرده‌سازی تا ۱۰ برابری را در ابعاد مدل‌های استاندارد پدید می‌آورد."""
)

# ==============================================================================
# Module 64: minimal_prism
# ==============================================================================
add_module(
    64,
    "minimal_prism",
    "فاکتورگیری طیف منفرد مبتنی بر بودجه انرژی",
    "MinimalPRISM Attention: Singular Spectrum Energy-Factorized Attention",
    r"""در شبکه‌های عمیق، لایه‌های پایانی برخلاف لایه‌های ابتدایی دارای طیف‌های مقادیر منفرد دم‌پهن هستند که نشان‌دهنده انتروپی اطلاعاتی بالای آن‌هاست. اعمال رتبه برشی کوچک در این لایه‌ها موجب از دست رفتن ویژگی‌های معنایی فرکانس بالا می‌شود. ماژول \lr{minimal\_prism} با توزیع متقارن جذر مقادیر منفرد و بودجه‌بندی هدایت‌شده، از پدیده فروپاشی رتبه در لایه‌های عمیق جلوگیری می‌کند.""",
    r"""برای لایه $l$، رتبه $r_l$ بر مبنای نسبت انرژی فروبنیوس تجمعی معین می‌گردد:
\begin{equation}
r_l = \arg\min_k \left\{ \frac{\sum_{i=1}^k \sigma_{l, i}^2}{\sum_{j=1}^p \sigma_{l, j}^2} \ge \eta_{\text{energy}} \right\}, \quad r_l \in [r_{\min}, r_{\max}]
\end{equation}
مقادیر منفرد به شکل متقارن بین ماتریس‌ها تقسیم می‌شوند:
\begin{equation}
\mathbf{W}_{\text{down}} = \operatorname{diag}\left(\sqrt{\mathbf{\Sigma}_{:r_l}}\right) \mathbf{V}_{:, :r_l}^\top, \quad \mathbf{W}_{\text{up}} = \mathbf{U}_{:, :r_l} \operatorname{diag}\left(\sqrt{\mathbf{\Sigma}_{:r_l}}\right)
\end{equation}
تبدیل ورودی با حفظ پایداری گرادیان به صورت $\mathbf{Y} = (\mathbf{X} \mathbf{W}_{\text{down}}^\top) \mathbf{W}_{\text{up}}^\top + \mathbf{b}$ ارزیابی می‌شود.""",
    r"""عملیات پروجکشن‌ها به $16 B N D r_l$ \lr{FLOP} تقلیل می‌یابد. تخصیص تطبیقی رتبه سبب توزیع متوازن بار حافظه در طول عمق مدل می‌گردد."""
)

# ==============================================================================
# Module 65: minimal_svd
# ==============================================================================
add_module(
    65,
    "minimal_svd",
    "توجه کم‌رتبه بر پایه بودجه انرژی فروبنیوس",
    "MinimalSVD: Cumulative Energy-Budgeted SVD Attention",
    r"""ماژول \lr{minimal\_svd} پیاده‌سازی بنیادی فاکتورگیری لایه‌های توجه در این پژوهش است. با پیاده‌سازی کلاس پایه \lr{FactorisedLinear}، تمامی ماتریس‌های پروجکشن با جفت ماتریس‌های کم‌رتبه جایگزین شده و گلوگاه دسترسی به حافظه در پردازنده‌های گرافیکی به شکل مؤثری دور زده می‌شود.""",
    r"""برای ماتریس اولیه $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ با انرژی فروبنیوس $\mathcal{E}_{\text{total}}$:
\begin{equation}
r = \min \left\{ k \;\Big|\; \frac{\sum_{i=1}^k \sigma_i^2}{\mathcal{E}_{\text{total}}} \ge E \right\}, \quad r \le 64
\end{equation}
\begin{equation}
\mathbf{A} = \mathbf{U}_{:, :r} \operatorname{diag}(\mathbf{\Sigma}_{:r}) \in \mathbb{R}^{d_{\text{out}} \times r}, \quad \mathbf{B} = \mathbf{V}_{:r, :}^\top \in \mathbb{R}^{r \times d_{\text{in}}}
\end{equation}
نگاشت برخط به فرم متوالی $\mathbf{Z} = \mathbf{X} \mathbf{B}^\top$ و سپس $\mathbf{Y} = \mathbf{Z} \mathbf{A}^\top + \mathbf{b}$ اجرا می‌گردد و خروجی آن به هدهای توجه استاندارد متصل می‌شود.""",
    r"""بار محاسباتی ضرب ماتریسی پروجکشن‌ها با ضریب $\frac{D}{2r}$ کاهش یافته و زمان تاخیر استنتاج در کارت‌های گرافیک با ظرفیت حافظه محدود بهبود چشمگیری می‌یابد."""
)

# ==============================================================================
# Module 66: minimal_svd_v2
# ==============================================================================
add_module(
    66,
    "minimal_svd_v2",
    "توجه کم‌رتبه SVD با مدیریت ماسک‌های ناهمگون",
    "MinimalSVD v2: Refined SVD Attention with Robust Mask Adaptation",
    r"""تنوع قالب‌های ماسک توجه در مدل‌های ترانسفورمر مختلف (مانند ماسک‌های دوبعدی پرکننده، سه‌بعدی علّی و چهاربعدی هدپایه)، در صورت عدم تطابق با لایه‌های فاکتورگیری‌شده، می‌تواند خطاهای پخش ابعادی یا تولید مقادیر نامعتبر (\lr{NaN}) به همراه داشته باشد. ماژول \lr{minimal\_svd\_v2} با تعبیه یک موتور مدیریت ماسک ناهمگون، پایداری عملیاتی را تضمین می‌نماید.""",
    r"""علاوه بر فاکتورگیری طیفی $\mathbf{W} \approx \mathbf{A} \mathbf{B}$، موتور مدیریت ماسک ابعاد را بازآرایی می‌کند:
\begin{equation}
\mathbf{M}_{\text{adapted}}[b, 1, i, j] = \begin{cases} 0.0 & \text{if valid token} \\ -10000.0 & \text{if padded token} \end{cases}
\end{equation}
پایداری عددی تابع \lr{Softmax} با تفریق مقدار بیشینه تضمین می‌شود:
\begin{equation}
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{adapted}}, \quad \mathbf{A} = \operatorname{Softmax}(\mathbf{S} - \max(\mathbf{S})), \quad \mathbf{Y} = \operatorname{FactorisedLinear}_O(\mathbf{A} \mathbf{V})
\end{equation}""",
    r"""پیچیدگی زمانی و مکانی همانند نسخه اول در سطح بهینه $\mathcal{O}(N D r)$ باقی مانده و خطاهای زمان اجرا در حین استنتاج روی معماری‌های متنوع به صفر می‌رسد."""
)

# ==============================================================================
# Module 67: mixattention_moska
# ==============================================================================
add_module(
    67,
    "mixattention_moska",
    "توجه تنک آگاه از واریانس فرکانسی FASA",
    "MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention",
    r"""بخش قابل توجهی از توکن‌های یک توالی بلند دارای اطلاعات معنایی تکراری یا نویز زمینه هستند. معماری \lr{mixattention\_moska} راهکار تنک‌سازی آگاه از واریانس فرکانسی (\lr{FASA}) را پیاده می‌سازد که انتروپی اطلاعاتی هر توکن را بر مبنای واریانس فعال‌سازی کانال‌های کلید ارزیابی می‌کند و با حفاظت قطعی از توکن لنگر $[\text{CLS}]$، نیمی از توکن‌ها را هرس می‌نماید.""",
    r"""برای تانسور کلید $\mathbf{K} \in \mathbb{R}^{B \times H \times L \times d_k}$، معیار اهمیت اطلاعاتی توکن $t$ برابر است با:
\begin{equation}
\operatorname{Importance}(t) = \frac{1}{H} \sum_{h=1}^H \operatorname{Var}_c(\mathbf{K}_{b, h, t})
\end{equation}
با لنگراندازی توکن ریشه‌ای $\operatorname{Importance}(0) \leftarrow 10^9$، تعداد $k = \lfloor L \times 0.5 \rfloor$ توکن برتر استخراج می‌شوند:
\begin{equation}
\mathcal{I}_{\text{sel}} = \operatorname{Top-}k(\operatorname{Importance}, k), \quad \mathbf{K}_p = \mathbf{K}[:, :, \mathcal{I}_{\text{sel}}, :], \quad \mathbf{V}_p = \mathbf{V}[:, :, \mathcal{I}_{\text{sel}}, :]
\end{equation}
توجه نامتقارن مقیاس‌شده به صورت زیر ارزیابی می‌گردد:
\begin{equation}
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}_p^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times L \times k}, \quad \mathbf{O} = \operatorname{Softmax}(\mathbf{S}) \mathbf{V}_p \in \mathbb{R}^{B \times H \times L \times d_k}
\end{equation}""",
    r"""ابعاد ماتریس شباهت از $L \times L$ به $L \times k$ منقبض می‌شود که مصرف حافظه لاجیت‌ها را تا ۷۵ درصد و زمان اجرای ضرب ماتریسی هسته توجه را تا ۵۰ درصد کاهش می‌دهد."""
)

# ==============================================================================
# Module 68: mla_attention
# ==============================================================================
add_module(
    68,
    "mla_attention",
    "فشرده‌سازی گلوگاه نهفته در DeepSeek-V2",
    "Multi-Head Latent Attention (MLA): Low-Rank Latent Bottleneck Compression",
    r"""در فاز رمزگشایی خودبازگشتی، حافظه نهان کلید و مقدار (\lr{KV-Cache}) گلوگاه اصلی مصرف حافظه گرافیکی و پهنای باند واکشی داده است. معماری انقلابی \lr{mla\_attention} (نوآوری محوری \lr{DeepSeek-V2}) با فشرده‌سازی حالت‌های کلید و مقدار در یک بردار نهفته کم‌رتبه $\mathbf{c}_{KV}$، حجم حافظه موقت را به شدت کاهش داده و با جذب ماتریس‌های پروجکشن در پرس‌وجو، راندمان استنتاج را دگرگون می‌سازد.""",
    r"""برای ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ و بعد گلوگاه نهفته $d_c \ll 2 H d_k$:
۱. فشرده‌سازی کلید-مقدار در بردار نهفته:
\begin{equation}
\mathbf{c}_{KV} = \mathbf{X} \mathbf{W}_{\text{DKV}} \in \mathbb{R}^{B \times N \times d_c}
\end{equation}
۲. بازسازی کلیدها و مقادیر: $\mathbf{K} = \mathbf{c}_{KV} \mathbf{W}_{\text{UK}}$ و $\mathbf{V} = \mathbf{c}_{KV} \mathbf{W}_{\text{UV}}$.
۳. سازوکار جذب ماتریسی در زمان استنتاج: در زمان تولید توکن، نیازی به نگهداری $\mathbf{K}$ نیست؛ ماتریس پروجکشن کلید در پرس‌وجو جذب می‌شود:
\begin{equation}
\mathbf{Q}_h \mathbf{K}_h^\top = (\mathbf{Q}_h \mathbf{W}_{\text{UK}, h}^\top) \mathbf{c}_{KV}^\top
\end{equation}
مشابهاً ماتریس مقدار در پروجکشن خروجی جذب می‌گردد و حافظه نهان تنها بردار کوچک $\mathbf{c}_{KV}$ را ذخیره می‌کند.""",
    r"""ردپای حافظه \lr{KV-Cache} به ازای هر توکن از $2 H d_k$ به $d_c$ عدد تقلیل می‌یابد که کاهشی معادل ۸۰ الی ۹۳ درصد را به ارمغان می‌آورد و ظرفیت اندازه دسته در پردازنده‌های گرافیکی را تا ۱۰ برابر افزایش می‌دهد."""
)

# ==============================================================================
# Module 69: moba
# ==============================================================================
add_module(
    69,
    "moba",
    "پوسته و تزریق‌کننده توجه مخلوطی از بلوک‌ها",
    "MoBA: Mixture-of-Block Attention Layer Wrapper & Registry Hook",
    r"""ماژول \lr{moba} به عنوان لایه واسط سطح سیستمی و تطبیق‌دهنده ساختاری برای مکانیزم توجه مخلوطی از بلوک‌ها (\lr{Mixture-of-Block Attention}) عمل می‌کند. هدف این ماژول، جداسازی تعریف واسط مدل از هسته محاسباتی، مدیریت پارامترهای پیکربندی اندازه بلوک، و تزریق یکپارچه لایه به مدل‌های مختلف با حفظ سازگاری گراف محاسباتی است.""",
    r"""این ماژول کلاس‌های \lr{MoBAAttention} و تابع تزریق \lr{inject\_moba} را بازصادرات می‌نماید. در هنگام جایگزینی در لایه‌های ترانسفورمر، پارامترهای زیر اعتبارسنجی و اعمال می‌شوند:
\begin{equation}
\mathcal{C} = \left\{ B_{\text{sz}} \in \{16, 32, 64\}, \; k \in \{2, 4, 8\}, \; \text{causal} \in \{\text{True}, \text{False}\}, \; \text{anchor\_cls} = \text{True} \right\}
\end{equation}
همگام‌سازی ابعاد در سراسر لایه‌ها بدون نیاز به بازآموزی اولیه تضمین می‌گردد.""",
    r"""سربار فراخوانی این لایه واسط در زمان اجرا $\mathcal{O}(1)$ بوده و دسترسی به اجرای سریع بلوکی را به صورت ماژولار در سراسر خط لوله میسر می‌سازد."""
)

# ==============================================================================
# Module 70: moba_attention
# ==============================================================================
add_module(
    70,
    "moba_attention",
    "توجه مخلوطی از بلوک‌ها با مسیریابی کسینوسی شبه‌مرکز",
    "MoBA Attention: Mixture-of-Block Attention via Centroid Cosine Routing",
    r"""پردازش متن‌های بسیار طولانی در ترانسفورمر استاندارد با سد محاسباتی مواجه است. ماژول \lr{moba\_attention} توالی را به بلوک‌های مجاور افراز کرده و با محاسبه بردار شبه‌مرکز (\lr{Centroid}) برای هر بلوک، یک مسیریابی دوسطحی پدید می‌آورد؛ هر بلوک پرس‌وجو تنها به $k$ بلوک برتر کلید متصل شده و محاسبات خرد‌دانه را فقط روی آن‌ها انجام می‌دهد.""",
    r"""توالی به $n_b = \lceil L / B_{\text{sz}} \rceil$ بلوک با اندازه $B_{\text{sz}}$ تقسیم می‌شود:
۱. محاسبه بردارهای شبه‌مرکز بلوک‌ها:
\begin{equation}
\bar{\mathbf{Q}}_i = \frac{1}{B_{\text{sz}}} \sum_{u=1}^{B_{\text{sz}}} \mathbf{Q}_{i, u}, \quad \bar{\mathbf{K}}_j = \frac{1}{B_{\text{sz}}} \sum_{v=1}^{B_{\text{sz}}} \mathbf{K}_{j, v} \quad \in \mathbb{R}^{B \times H \times d_k}
\end{equation}
۲. ماتریس شباهت بین‌بلوکی: $\mathbf{S}_{i, j}^{\text{block}} = \frac{\bar{\mathbf{Q}}_i \bar{\mathbf{K}}_j^\top}{\sqrt{d_k}}$.
۳. انتخاب تنک $k$ بلوک برتر به علاوه بلوک قطری (توجه به خود): $\mathcal{I}_i = \operatorname{Top-}k(\mathbf{S}_{i, :}^{\text{block}}) \cup \{i\}$.
۴. محاسبه توجه دقیق خرد‌دانه:
\begin{equation}
\mathbf{O}_i = \operatorname{Softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_{\text{sel}, i}^\top}{\sqrt{d_k}} + \mathbf{M}_i\right) \mathbf{V}_{\text{sel}, i} \in \mathbb{R}^{B_{\text{sz}} \times d_k}
\end{equation}""",
    r"""پیچیدگی محاسباتی توجه از $\mathcal{O}(L^2 D)$ به $\mathcal{O}(L \cdot k B_{\text{sz}} \cdot D)$ نزول می‌کند که به دلیل $k B_{\text{sz}} \ll L$ رفتاری کاملاً خطی با طول دنباله را رقم می‌زند."""
)

# ==============================================================================
# Module 71: moe_deltakv_paged
# ==============================================================================
add_module(
    71,
    "moe_deltakv_paged",
    "توجه متخصصان با حافظه کوانتیزه‌شده و صفحه‌بندی‌شده",
    "MoE-DeltaKV Paged: Quantized & Paged KV-Cache Mixture-of-Experts Attention",
    r"""در استنتاج طولانی‌مدت، رشد حافظه موقت و گسستگی حافظه فیزیکی (\lr{Memory Fragmentation}) راندمان را کاهش می‌دهند. ماژول \lr{moe\_deltakv\_paged} سه نوآوری مکمل را در هم می‌آمیزد: فشرده‌سازی تفاضلی زمان‌محور کلیدها با \lr{SVD}، کوانتیزاسیون ۴ بیتی یکنواخت با تخمین‌گر گرادیان مستقیم (\lr{STE})، و مدیریت حافظه صفحه‌بندی‌شده (\lr{Paged Memory}) با هدایت لنگری توکن‌های حیاتی.""",
    r"""۱. تقریب کم‌رتبه تفاضل زمانی کلیدها: $\Delta \mathbf{K} = \mathbf{K}_t - \mathbf{K}_{t-1} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^\top$.
۲. کوانتیزاسیون ۴ بیتی با عملگر \lr{STE}:
\begin{equation}
\operatorname{STE-Quant}(\mathbf{x}) = \mathbf{x} + \left(\operatorname{clamp}\left(\operatorname{round}\left(\frac{\mathbf{x} - x_{\min}}{\Delta}\right), 0, 15\right) \cdot \Delta + x_{\min} - \mathbf{x}\right).\operatorname{detach}()
\end{equation}
مشتق معکوس عملگر همانی بوده ($\frac{\partial \operatorname{STE-Quant}(\mathbf{x})}{\partial \mathbf{x}} = \mathbf{I}$) و جریان گرادیان مسدود نمی‌شود.
۳. تخصیص در صفحات فیزیکی ثابت با اندازه ۶۴ با گنجاندن همیشگی صفحه صفر برای حفظ زمینه پرامپت.""",
    r"""ردپای حافظه \lr{KV-Cache} تا ۷۵ درصد منقبض شده و پیچیدگی زمانی با تابع $\mathcal{O}\left(k N D + \frac{N^2}{B_s^2} d_k\right)$ مقیاس‌بندی خطی را حفظ می‌نماید."""
)

# ==============================================================================
# Module 72: nasc_attention
# ==============================================================================
add_module(
    72,
    "nasc_attention",
    "آبشار طیفی NEXUS-AEGIS برای پردازش خطی",
    "NASC: NEXUS-AEGIS Spectral Cascade Attention",
    r"""معماری \lr{nasc\_attention} یک ساختار آبشاری سه‌سطحی است که سرعت پردازش بلوکی را با دقت بالای فعال‌سازهای اصلاح‌شده و بازیابی سراسری در هم می‌آمیزد: ابتدا توالی به بلوک‌های اندازه ۳۲ افراز شده و امتیازدهی درشت‌دانه صورت می‌گیرد؛ سپس درون بلوک‌های فعال، فعال‌ساز غیرخطی \lr{Softpick} وزن‌های نویز را به صفر دقیق می‌رساند؛ در پایان، یک مسیر طیفی تصادفی زمینه دوربرد را در زمان خطی پوشش می‌دهد.""",
    r"""۱. امتیازدهی بلوکی با شبه‌مرکزها: $\mathbf{S}_{ij}^{\text{blk}} = \frac{\bar{\mathbf{Q}}_i \bar{\mathbf{K}}_j^\top}{\sqrt{d_k}}$ و انتخاب $\mathcal{I}_i = \operatorname{Top-}k(\mathbf{S}_{i, :}^{\text{blk}}, k)$.
۲. ارزیابی درون‌بلوکی با فعال‌ساز اصلاح‌شده \lr{Softpick}:
\begin{equation}
\mathbf{O}_{\text{sp}, i} = \operatorname{Softpick}\left(\frac{\mathbf{Q}_i \mathbf{K}_{\mathcal{I}_i}^\top}{\sqrt{d_k}}\right) \mathbf{V}_{\mathcal{I}_i}
\end{equation}
\begin{equation}
\operatorname{Softpick}(\mathbf{x})_u = \frac{\operatorname{ReLU}\left(\exp\left(\frac{x_u - m}{T}\right) - \exp\left(\frac{-m}{T}\right)\right)}{\sum_v \operatorname{ReLU}\left(\exp\left(\frac{x_v - m}{T}\right) - \exp\left(\frac{-m}{T}\right)\right) + \epsilon}
\end{equation}
۳. تصحیح طیفی خطی و تلفیق با گیتینگ: $\mathbf{Y} = \alpha \mathbf{O}_{\text{sp}} + (1 - \alpha) \mathbf{O}_{\text{spectral}}$.""",
    r"""این معماری با پیچیدگی زمانی و مکانی اکیداً خطی $\mathcal{O}(L)$ اجرا شده و عملکردی پایدار در طول دنباله‌های فراتر از ۴۰۹۶ توکن ارائه می‌دهد."""
)

# ==============================================================================
# Module 73: nexus
# ==============================================================================
add_module(
    73,
    "nexus",
    "توجه تنک یکپارچه با امتیازدهی ضربدری و بازیافت طیفی",
    "NEXUS: Neural Efficient X-scoring Unified Sparse Attention",
    r"""مکانیزم \lr{nexus} یک معماری چندمقیاسی فوق‌پیشرفته است که چهار ستون نظری را ادغام می‌کند: امتیازدهی ضدقطری بلوک‌ها با گام‌های متناوب (\lr{XAttention})، گیتینگ هدهای یادگیرنده (\lr{SeerAttention})، توجه نواری با تابع \lr{Softpick}، و تصحیح خطی دوره‌ای با نگاشت تصادفی فوریه (\lr{Spectraformer RFF}). این ترکیب ساختاریافته علاوه بر کاهش محاسبات، معضل توکن‌های فرورونده (\lr{Attention Sinks}) را که انرژی توجه را به هدر می‌دهند مهار می‌سازد.""",
    r"""۱. امتیازدهی بلوکی با گام زیرنمونه‌گیری $\sigma = 2$:
\begin{equation}
\mathbf{S}_{b_i, b_j} = \frac{1}{(B_s / \sigma)^2} \sum_{u=0}^{B_s/\sigma - 1} \sum_{v=0}^{B_s/\sigma - 1} \frac{\mathbf{q}_{b_i, u} \mathbf{k}_{b_j, v}^\top}{\sqrt{D}}
\end{equation}
۲. پیش‌بینی پویای بودجه بلوک‌ها در هر هد توسط \lr{SeerAttention}:
\begin{equation}
g_h = \sigma\left(\frac{\operatorname{MLP}_q(\bar{\mathbf{q}}_h) \operatorname{MLP}_k(\bar{\mathbf{k}}_h)^\top}{T_{\text{seer}}}\right), \quad k_h = \operatorname{clamp}\left(\lfloor g_h \cdot M \rfloor, 1, M\right)
\end{equation}
۳. قضیه تنکی صفر مطلق با تابع \lr{Softpick}: برای هر لاجیت $x_i \le 0$ با نقطه اتکای میانگین $m$:
\begin{equation}
\operatorname{Softpick}(\mathbf{x})_i = \frac{\operatorname{ReLU}\left(\exp\left(\frac{x_i - m}{T}\right) - \exp\left(\frac{-m}{T}\right)\right)}{\sum_j \operatorname{ReLU}\left(\exp\left(\frac{x_j - m}{T}\right) - \exp\left(\frac{-m}{T}\right)\right) + \epsilon} = 0
\end{equation}
۴. تصحیح طیفی دوره‌ای با بردار تصادفی نرمال‌شده $\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \exp\left(\operatorname{clamp}\left(\frac{\mathbf{x}}{\|\mathbf{x}\|_2} \mathbf{W}_{\text{rff}}, -\infty, 11\right)\right)$.""",
    r"""پیچیدگی زمانی به $\mathcal{O}(N \cdot k_{\text{avg}} \cdot B_s \cdot D)$ تقلیل یافته و حافظه فعال‌سازی در طول توالی‌های طولانی تا بیش از ۸۵ درصد کاهش می‌یابد."""
)

# ==============================================================================
# Module 74: nha_attention
# ==============================================================================
add_module(
    74,
    "nha_attention",
    "توجه ترکیبی پنجره محلی و زمینه سراسری",
    "Native Hybrid Attention (NHA): Local Window & Global Downsampling",
    r"""روابط نحوی و دستوری در متون بیشتر میان واژگان همسایه در یک پنجره موضعی کوچک رخ می‌دهند، در حالی که اطلاعات معنایی نیازمند دید سراسری با وضوح پایین‌تر هستند. ماژول \lr{nha\_attention} این دو مولفه را در دو شاخه مجزا تفکیک می‌کند: شاخه موضعی با پنجره لغزان دقیق و شاخه سراسری با زیرنمونه‌گیری گام‌دار با پیچیدگی خطی.""",
    r"""۱. شاخه موضعی با پنجره لغزان نواری به شعاع $W_{\text{local}}$:
\begin{equation}
\mathbf{M}_{\text{local}}[i, j] = \begin{cases} 0 & \text{if } |i - j| \le W_{\text{local}} \\ -\infty & \text{otherwise} \end{cases}
\end{equation}
\begin{equation}
\mathbf{O}_{\text{local}} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{local}}\right) \mathbf{V}
\end{equation}
۲. شاخه سراسری با فشرده‌سازی مقیاس توالی با گام $s$:
\begin{equation}
\mathbf{K}_{\text{global}} = \operatorname{Downsample}(\mathbf{K}, \text{stride}=s), \quad \mathbf{V}_{\text{global}} = \operatorname{Downsample}(\mathbf{V}, \text{stride}=s)
\end{equation}
\begin{equation}
\mathbf{O}_{\text{global}} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}_{\text{global}}^\top}{\sqrt{d_k}}\right) \mathbf{V}_{\text{global}}
\end{equation}
۳. ادغام شاخه‌ها با ضریب تلفیق یادگیرنده: $\mathbf{Y} = \lambda \odot \mathbf{O}_{\text{local}} + (1 - \lambda) \odot \mathbf{O}_{\text{global}}$.""",
    r"""پیچیدگی مسیر محلی $\mathcal{O}(L \cdot W_{\text{local}} \cdot D)$ و مسیر سراسری $\mathcal{O}(L \cdot (L/s) \cdot D)$ است که بار محاسباتی را بیش از ۷۰ درصد نسبت به حالت چگال کاهش می‌دهد."""
)

# ==============================================================================
# Module 75: nha_compressor
# ==============================================================================
add_module(
    75,
    "nha_compressor",
    "فشرده‌ساز توالی مبتنی بر توجه ترکیبی NHA",
    "NHA Compressor: Sequence Downsampling Adapter Engine",
    r"""ماژول \lr{nha\_compressor} به عنوان موتور اختصاصی فشرده‌سازی طول توالی در رجیستری فشرده‌سازها (\lr{COMPRESSION\_REGISTRY}) ثبت شده است. این ماژول وظیفه فشرده‌سازی بعد توالی ماتریس‌های کلید و مقدار را در لایه‌های میانی بر عهده دارد تا از پردازش دنباله‌های طویل در لایه‌های عمیق‌تر جلوگیری نماید.""",
    r"""این ماژول طول توالی $L$ را از طریق حوضچه‌بندی سازگار به طول فشرده $L_{\text{comp}} \ll L$ منقبض می‌نماید:
\begin{equation}
\mathbf{K}_{\text{compressed}} = \operatorname{NHACompressor}(\mathbf{K}), \quad \mathbf{V}_{\text{compressed}} = \operatorname{NHACompressor}(\mathbf{V}) \in \mathbb{R}^{B \times H \times L_{\text{comp}} \times d_k}
\end{equation}
توکن کلاسه‌بندی در شاخص صفر با لنگراندازی قطعی حفظ می‌شود و لایه‌های بعدی مدل به جای توالی کامل روی بردارهای متراکم عمل می‌نمایند.""",
    r"""مصرف حافظه فعال‌سازی در لایه‌های متعاقب با نسبت $\frac{L_{\text{comp}}}{L}$ منقبض شده و پردازش کلی مدل شتاب می‌گیرد."""
)

# ==============================================================================
# Module 76: nsa_attention
# ==============================================================================
add_module(
    76,
    "nsa_attention",
    "توجه تنک بومی با پنجره لغزان و مسیریابی پویا",
    "Native Sparse Attention (NSA): Sliding-Window & Dynamic Top-k Routing",
    r"""معماری \lr{nsa\_attention} یک مکانیزم تنک‌سازی دوگانه و یکپارچه است که بدون نیاز به شبکه‌های راهنمای جداگانه، همبستگی‌های نزدیک را با پنجره لغزان ایستا و روابط بلندمدت را با مسیریابی پویای بلوک‌های برتر ثبت می‌کند.""",
    r"""۱. ماسک نواری استاتیک موضعی با شعاع $W_{\text{window}}$:
\begin{equation}
\mathbf{M}_{\text{local}}[i, j] = \begin{cases} 0 & \text{if } |i - j| \le W_{\text{window}} \\ -\infty & \text{otherwise} \end{cases}
\end{equation}
۲. ارزیابی امتیاز درشت‌دانه بلوک‌ها با اندازه $B_s$:
\begin{equation}
\mathbf{S}_{b_i, b_j}^{\text{block}} = \frac{\bar{\mathbf{Q}}_{b_i} \bar{\mathbf{K}}_{b_j}^\top}{\sqrt{d_k}}, \quad \mathcal{B}_i = \operatorname{Top-}k\left(\mathbf{S}_{b_i, :}^{\text{block}}\right)
\end{equation}
۳. ماسک تنک تلفیقی: $\mathbf{M}_{\text{sparse}}[i, j] = 0$ در صورتی که $|i - j| \le W_{\text{window}}$ یا $\operatorname{block}(j) \in \mathcal{B}_{\operatorname{block}(i)}$ باشد.
\begin{equation}
\mathbf{O} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{sparse}}\right) \mathbf{V}
\end{equation}""",
    r"""تعداد درایه‌های فعال ماتریس توجه به $L (2 W_{\text{window}} + k B_s)$ کاهش یافته و پیچیدگی زمانی و مکانی $\mathcal{O}(L \cdot d_k)$ کاملاً خطی می‌گردد."""
)

# ==============================================================================
# Module 77: nystrom_attention
# ==============================================================================
add_module(
    77,
    "nystrom_attention",
    "تقریب رتبه پایین با شبه‌معکوس نایستروم",
    "Nyström Attention: Low-Rank Landmark Approximation via Matrix Pseudoinverse",
    r"""تئوری کلاسیک نایستروم تقریب ماتریس‌های هسته با ابعاد بزرگ را از طریق انتخاب زیرمجموعه‌ای از سطرها و ستون‌ها میسر می‌سازد. ماژول \lr{nystrom\_attention} از این قضیه برای تجزیه ماتریس توجه به سه زیرماتریس کوچک با استفاده از $m \ll L$ توکن شاخص (\lr{Landmarks}) بهره می‌گیرد و با حل عددی پایدار شبه‌معکوس با روش نیوتن-شولتز، پیچیدگی را خطی می‌سازد.""",
    r"""۱. استخراج $m$ نقطه شاخص با میانگین‌گیری قطعات هم‌اندازه توالی:
\begin{equation}
\widetilde{\mathbf{Q}}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{Q}_i, \quad \widetilde{\mathbf{K}}_k = \frac{1}{|S_k|} \sum_{j \in S_k} \mathbf{K}_j, \quad k \in \{1, \dots, m\}
\end{equation}
۲. زیرماتریس‌های سه‌گانه هسته:
\begin{equation}
\widehat{\mathbf{F}}_1 = \operatorname{Softmax}\left(\frac{\mathbf{Q} \widetilde{\mathbf{K}}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times m}, \quad \widehat{\mathbf{F}}_2 = \operatorname{Softmax}\left(\frac{\widetilde{\mathbf{Q}} \widetilde{\mathbf{K}}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times m}, \quad \widehat{\mathbf{F}}_3 = \operatorname{Softmax}\left(\frac{\widetilde{\mathbf{Q}} \mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times L}
\end{equation}
۳. محاسبه تکراری شبه‌معکوس با روش نیوتن-شولتز با شرط آغازین مقیاس‌شده:
\begin{equation}
\mathbf{Z}_0 = \frac{\widehat{\mathbf{F}}_2^\top}{\|\widehat{\mathbf{F}}_2\|_1 \|\widehat{\mathbf{F}}_2\|_\infty}, \quad \mathbf{Z}_{t+1} = \mathbf{Z}_t \left( 2 \mathbf{I}_m - \widehat{\mathbf{F}}_2 \mathbf{Z}_t \right)
\end{equation}
۴. تجمیع شرکت‌پذیر از راست به چپ: $\operatorname{Context} = \widehat{\mathbf{F}}_1 \left( \mathbf{Z}_T \left( \widehat{\mathbf{F}}_3 \mathbf{V} \right) \right) \in \mathbb{R}^{L \times d_k}$.""",
    r"""پیچیدگی محاسباتی برابر با $\mathcal{O}(L \cdot m \cdot d_k + m^3)$ است که نسبت به طول دنباله $L$ اکیداً خطی بوده و بن‌بست حافظه را در توالی‌های بسیار طولانی مرتفع می‌کند."""
)

# ==============================================================================
# Module 78: optimized_hybrid_attention
# ==============================================================================
add_module(
    78,
    "optimized_hybrid_attention",
    "توجه ترکیبی پنجره لغزان محلی و فشرده‌سازی سراسری",
    "Optimized Hybrid Attention (OHA): Sliding-Window & Latent Sequence Compression",
    r"""معماری \lr{optimized\_hybrid\_attention} تعادلی کارآمد میان تحلیل نحوی موضعی و آگاهی سراسری متن برقرار می‌سازد. در این ماژول، شاخه موضعی بر روابط نزدیک تمرکز دارد و شاخه سراسری با ماتریس‌های یادگیرنده فشرده‌سازی بعد توالی، طول دنباله را به یک بعد گلوگاه کوچک منقبض می‌نماید.""",
    r"""۱. شاخه موضعی با پنجره لغزان قطری به شعاع $W_{\text{window}}$:
\begin{equation}
\mathbf{O}_{\text{local}} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{window}}\right) \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_k}
\end{equation}
۲. شاخه سراسری با ماتریس‌های یادگیرنده فشرده‌سازی بعد توالی $\mathbf{P}_K, \mathbf{P}_V \in \mathbb{R}^{R \times N}$ ($R \ll N$):
\begin{equation}
\widetilde{\mathbf{K}} = \operatorname{Softmax}(\mathbf{P}_K, \text{dim}=-1) \mathbf{K}, \quad \widetilde{\mathbf{V}} = \operatorname{Softmax}(\mathbf{P}_V, \text{dim}=-1) \mathbf{V} \quad \in \mathbb{R}^{B \times H \times R \times d_k}
\end{equation}
\begin{equation}
\mathbf{O}_{\text{global}} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \widetilde{\mathbf{K}}^\top}{\sqrt{d_k}}\right) \widetilde{\mathbf{V}} \in \mathbb{R}^{B \times H \times N \times d_k}
\end{equation}
۳. ادغام شاخه‌ها با گیت هدپایه یادگیرنده $\alpha \in \mathbb{R}^H$: $\mathbf{O}_h = \sigma(\alpha_h) \mathbf{O}_{\text{local}, h} + (1 - \sigma(\alpha_h)) \mathbf{O}_{\text{global}, h}$.""",
    r"""مجموع پیچیدگی با تابع $\mathcal{O}(N (W_{\text{window}} + R) D)$ توصیف می‌شود که کاملاً نسبت به طول توالی خطی است."""
)

# ==============================================================================
# Module 79: orion
# ==============================================================================
add_module(
    79,
    "orion",
    "شبکه متعامد تطبیقی با فاکتورگیری SVD و توجه دوشاخه‌ای",
    "ORION: Input-Aware Orthogonal Network with SVD Projections & Bifurcated Attention",
    r"""معماری \lr{orion} فشرده‌سازی رتبه-پایین پارامترها را با توجه تنک دوشاخه‌ای تلفیق می‌کند: در لایه پروجکشن، ماتریس‌ها توسط \lr{SVD} با حفظ ۹۵ درصد انرژی فاکتورگیری می‌شوند؛ سپس هدهای توجه به دو گروه مستقل تفکیک می‌گردند: هدهای چگال با پنجره موضعی و فعال‌ساز اصلاح‌شده \lr{Softpick} و هدهای هسته‌ای با ویژگی‌های تصادفی فوریه (\lr{RFF}) متعامد.""",
    r"""۱. پروجکشن‌های فاکتورگیری‌شده: $\mathbf{W} \approx \mathbf{W}_{\text{up}} \mathbf{W}_{\text{down}}$ با رتبه $r \ll D$.
۲. تفکیک هدهای توجه به دو دسته $H_{\text{dense}}$ و $H_{\text{kernel}}$ ($H = H_{\text{dense}} + H_{\text{kernel}}$):
برای هدهای چگال موضعی:
\begin{equation}
\mathbf{S}_{\text{dense}} = \frac{\mathbf{Q}_d \mathbf{K}_d^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{window}}, \quad \mathbf{O}_{\text{dense}} = \operatorname{Softpick}(\mathbf{S}_{\text{dense}}) \mathbf{V}_d
\end{equation}
برای هدهای هسته‌ای خطی با ویژگی‌های تصادفی مثبت:
\begin{equation}
\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} \exp\left(\frac{\mathbf{x} \mathbf{\Omega}^\top}{\sqrt{d_k}} - \frac{\|\mathbf{x}\|_2^2}{2 d_k}\right), \quad \mathbf{O}_{\text{kernel}} = \frac{\phi(\mathbf{Q}_k) \left( \phi(\mathbf{K}_k)^\top \mathbf{V}_k \right)}{\phi(\mathbf{Q}_k) \left( \phi(\mathbf{K}_k)^\top \mathbf{1}_N \right) + \epsilon}
\end{equation}
۳. ادغام خروجی با نگاشت فاکتورگیری‌شده نهایی صورت می‌پذیرد.""",
    r"""فاکتورگیری وزن‌ها حجم پارامترها را تا ۶۰ درصد و محاسبات توجه خطی با مرتبه $\mathcal{O}(N (W_{\text{window}} + m) D)$ بار پردازش را به شدت تقلیل می‌دهند."""
)

# ==============================================================================
# Module 80: performer_attention
# ==============================================================================
add_module(
    80,
    "performer_attention",
    "توجه خطی با ویژگی‌های تصادفی متعامد مثبت FAVOR+",
    "Performer Attention (FAVOR+): Fast Attention via Positive Orthogonal Random Features",
    r"""تلاش‌های پیشین برای خطی‌سازی توجه با استفاده از ویژگی‌های مثلثاتی تصادفی فوریه (\lr{RFF}) به دلیل تولید مقادیر منفی دچار ناپایداری و واگرایی شدید می‌شدند. ماژول \lr{performer\_attention} چارچوب اثبات‌شده \lr{FAVOR+} را معرفی می‌کند که با نگاشت پرس‌وجوها و کلیدها به یک فضای ویژگی تصادفی اکیداً مثبت و بهره‌گیری از خاصیت شرکت‌پذیری، توجه خطی نااریب و کاملاً پایداری را ارائه می‌دهد.""",
    r"""۱. فرمول‌بندی نگاشت تصادفی مثبت \lr{FAVOR+}:
\begin{equation}
\phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{m}} \exp\left(\mathbf{x} \mathbf{\Omega}^\top - \frac{\|\mathbf{x}\|_2^2}{2}\right) \quad \text{یا} \quad \phi(\mathbf{x}) = \frac{\operatorname{ReLU}(\mathbf{x} \mathbf{\Omega}^\top) + \epsilon}{\sqrt{m} + \epsilon}
\end{equation}
۲. اثبات کاهش واریانس با ویژگی‌های تصادفی متعامد (\lr{ORF}):
با متعامدسازی سطرهای $\mathbf{\Omega}$ با روش گرام-اشمیت ($\mathbf{\omega}_i \perp \mathbf{\omega}_j$)، کوواریانس بین برآوردگرها منفی شده ($\operatorname{Cov}_{\text{ORF}}(Z_i, Z_j) \le 0$) و واریانس کل کاهش می‌یابد:
\begin{equation}
\operatorname{Var}\left(\hat{K}_{\text{ORF}}\right) \le \operatorname{Var}\left(\hat{K}_{\text{IID}}\right)
\end{equation}
۳. انقباض ماتریسی خطی:
\begin{equation}
\mathbf{M}_{KV} = \phi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{m \times d_v}, \quad \mathbf{D}_{\text{sum}} = \phi(\mathbf{K})^\top \mathbf{1}_L \in \mathbb{R}^{m}
\end{equation}
\begin{equation}
\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \frac{\phi(\mathbf{Q}) \mathbf{M}_{KV}}{\phi(\mathbf{Q}) \mathbf{D}_{\text{sum}} + \epsilon} \in \mathbb{R}^{L \times d_v}
\end{equation}""",
    r"""پیچیدگی زمانی $\mathcal{O}(L \cdot m \cdot d)$ و حافظه $\mathcal{O}(L \cdot m + m \cdot d)$ کاملاً خطی بوده و امکان اجرای بلادرنگ روی دنباله‌های ۳۲ هزار توکنی را فراهم می‌سازد."""
)

# ==============================================================================
# Module 81: phantom
# ==============================================================================
add_module(
    81,
    "phantom",
    "توجه درهم‌سازی‌شده با تجزیه CUR و نگاشت متعامد",
    "PHANTOM: Deterministic CUR Matrix Decomposition & Multi-Round Angular LSH",
    r"""معماری \lr{phantom} ساختاری تلفیقی برای دستیابی همزمان به تنکی بالا و فشرده‌سازی پارامتری است. این معماری تجزیه ماتریسی آماری \lr{CUR} (بر پایه امتیازهای اهرمی آماری \lr{Leverage Scores}) را با درهم‌سازی زاویه‌ای حساس به مکان چند دوره‌ای و فعال‌ساز اصلاح‌شده \lr{Softpick} ترکیب می‌کند تا علاوه بر فشرده‌سازی لایه‌های وزن، بار توجه روی توکن‌ها را به حداقل برساند.""",
    r"""۱. تجزیه قطعی \lr{CUR} مبتنی بر امتیازهای اهرمی:
برای ماتریس وزن $\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$:
\begin{equation}
\pi_i^{(\text{row})} = \frac{1}{r} \|\mathbf{U}_{i, :r}\|_2^2, \quad \pi_j^{(\text{col})} = \frac{1}{r} \|\mathbf{V}_{j, :r}\|_2^2
\end{equation}
با انتخاب $r$ سطر ($\mathbf{R}$) و ستون ($\mathbf{C}$) برتر، فاکتورگیری به فرم $\mathbf{W} \approx \mathbf{C} \mathbf{W}_\cap^+ \mathbf{R}$ انجام می‌پذیرد.
۲. درهم‌سازی چند دوره‌ای زاویه‌ای با ابرصفحه‌های تصادفی $\mathbf{P}$:
\begin{equation}
h(\mathbf{x}) = \arg\max \left[ \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \mathbf{P} \;;\; -\frac{\mathbf{x}}{\|\mathbf{x}\|_2} \mathbf{P} \right]
\end{equation}
۳. توجه اصلاح‌شده درون‌سطل با تابع \lr{Softpick}: $\mathbf{A}_{ij}^{(m)} = \operatorname{Softpick}\left(\frac{\mathbf{q}_i \mathbf{k}_j^\top}{\sqrt{d_k}}\right) \cdot \mathbb{I}(h(\mathbf{q}_i) = h(\mathbf{k}_j))$.""",
    r"""محاسبات پروجکشن با تجزیه \lr{CUR} به $16 B N D r$ تقلیل یافته و هزینه توجه درون‌سطل در مقیاس $\mathcal{O}\left(M \cdot B \cdot D \cdot \frac{N^2}{2^{B_s}}\right)$ ارزیابی می‌شود."""
)

# ==============================================================================
# Module 82: primal_attention
# ==============================================================================
add_module(
    82,
    "primal_attention",
    "تقریب اولیه هسته SVD با ادغام بندهای توالی",
    "Primal Attention: Asymmetric Kernel SVD via Segment-Pooled Landmarks",
    r"""معماری \lr{primal\_attention} تقریب هسته را در فضای اولیه بازنمایی (\lr{Primal Space}) شکل می‌دهد. این ماژول $r \ll L$ لنگرگاه شاخص را با ادغام یکنواخت توالی استخراج کرده و ماتریس توجه نامتقارن را به سه زیرماتریس تجزیه می‌کند؛ سپس با محاسبه شبه‌معکوس ماتریس میانی از طریق الگوریتم شولتز، تجمیع در زمان خطی انجام می‌گیرد.""",
    r"""۱. استخراج لنگرگاه‌ها با حوضچه‌بندی میانگین یک‌بعدی:
\begin{equation}
\widetilde{\mathbf{Q}}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{Q}_i, \quad \widetilde{\mathbf{K}}_k = \frac{1}{|S_k|} \sum_{j \in S_k} \mathbf{K}_j \quad \in \mathbb{R}^{r \times d}
\end{equation}
۲. زیرماتریس‌های نامتقارن هسته:
\begin{equation}
\mathbf{F}_1 = \operatorname{Softmax}\left(\frac{\mathbf{Q} \widetilde{\mathbf{K}}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{L \times r}, \quad \mathbf{F}_2 = \operatorname{Softmax}\left(\frac{\widetilde{\mathbf{Q}} \widetilde{\mathbf{K}}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{r \times r}, \quad \mathbf{F}_3 = \operatorname{Softmax}\left(\frac{\widetilde{\mathbf{Q}} \mathbf{K}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{r \times L}
\end{equation}
۳. محاسبه پایدار شبه‌معکوس $\mathbf{F}_2^+$ با تکرار نیوتن-شولتز: $\mathbf{Z}_{t+1} = \mathbf{Z}_t (2 \mathbf{I}_r - \mathbf{F}_2 \mathbf{Z}_t)$.
۴. تجمیع زمینه اولیه از راست به چپ: $\operatorname{Context} = \mathbf{F}_1 \left( \mathbf{F}_2^+ \left( \mathbf{F}_3 \mathbf{V} \right) \right) \in \mathbb{R}^{L \times d}$.""",
    r"""پیچیدگی محاسباتی $\mathcal{O}(L \cdot r \cdot d + r^3)$ اکیداً خطی با طول دنباله $L$ است و نیاز به ایجاد ماتریس مربعی $L \times L$ را کاملاً منتفی می‌سازد."""
)

# ==============================================================================
# Module 83: progressive_token_pruning
# ==============================================================================
add_module(
    83,
    "progressive_token_pruning",
    "هرس تدریجی و لایه‌به‌لایه‌ی توکن‌ها",
    "Progressive Token Pruning (PTP): Dynamic Layer-Wise Token Elimination",
    r"""در لایه‌های عمیق مدل‌های زبانی، بسیاری از توکن‌ها به اشباع معنایی می‌رسند و نگه‌داری آن‌ها بار پردازشی بی‌ثمری ایجاد می‌کند. مکانیزم \lr{progressive\_token\_pruning} یک سیاست هرس تدریجی را اعمال می‌کند که طول دنباله را لایه‌به‌لایه به شکل یکنواخت کاهش می‌دهد، در حالی که توکن حیاتی $[\text{CLS}]$ همواره حفظ می‌گردد.""",
    r"""۱. محاسبه شاخص اهمیت توکن در لایه $l$:
\begin{equation}
\operatorname{Score}(t) = \|\mathbf{X}_{l, t}\|_2 \quad \text{یا} \quad \operatorname{Score}(t) = \frac{1}{H} \sum_{h=1}^H \sum_{j=1}^{N_l} \mathbf{A}_{h, j, t}
\end{equation}
۲. حفاظت نامحدود از توکن ریشه‌ای: $\operatorname{Score}(0) \leftarrow +\infty$.
۳. جدول زمانی نگه‌داشت توکن‌ها در طول لایه‌های $l \in \{1, \dots, L_{\text{layers}}\}$:
\begin{equation}
\rho_l = 1 - \beta \frac{l}{L_{\text{layers}}}, \quad k_l = \max\left(k_{\min}, \; \lfloor N_0 \cdot \rho_l \rfloor\right)
\end{equation}
۴. استخراج زیرمجموعه برای لایه بعدی: $\mathbf{X}_{l+1} = \mathbf{X}_l[\operatorname{Top-}k_l(\operatorname{Score}), :] \in \mathbb{R}^{B \times k_l \times D}$.""",
    r"""پیچیدگی محاسباتی به مجموع کاهشی $\sum_{l=1}^{L_{\text{layers}}} \mathcal{O}(k_l^2 D)$ تبدیل می‌شود که در مقایسه با حالت پایه بیش از ۴۵ الی ۶۰ درصد کاهش در کل \lr{FLOP}های مدل ایجاد می‌کند."""
)

# ==============================================================================
# Module 84: projection_factorized_attn
# ==============================================================================
add_module(
    84,
    "projection_factorized_attn",
    "توجه خطی با فاکتورگیری نگاشت SVD",
    "Projection Factorized SVD Linear Attention: Dual-Level Factorization Architecture",
    r"""ماژول \lr{projection\_factorized\_attn} (زیرکلاس تخصصی \lr{SVDLinearAttention}) یک چارچوب دوگانه برای فشرده‌سازی پارامترها و خطی‌سازی محاسبات است. این معماری همزمان دو تنگنای اساسی ترانسفورمر را برطرف می‌کند: در سطح پارامتری با تجزیه کم‌رتبه ماتریس‌های نگاشت پروجکشن بر پایه \lr{SVD} حافظه مدل را مهار می‌سازد، و در سطح دنباله با نگاشت‌های هسته‌ای غیرخطی مثبت به جای تابع سافت‌مکس، محاسبات ماتریسی توجه را به زمان خطی $\mathcal{O}(N)$ تبدیل می‌کند.""",
    r"""۱. فاکتورگیری کم‌رتبه ماتریس‌های وزن پروجکشن ورودی:
\begin{equation}
\mathbf{W} \approx \mathbf{W}_{\text{up}} \mathbf{W}_{\text{down}}, \quad \mathbf{W}_{\text{down}} \in \mathbb{R}^{r \times D}, \quad \mathbf{W}_{\text{up}} \in \mathbb{R}^{D \times r}
\end{equation}
۲. خطی‌سازی توجه با توابع ویژگی هسته‌ای مثبت: $\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1$.
۳. انقباض ماتریسی شرکت‌پذیر در زمان خطی:
\begin{equation}
\mathbf{S}_{KV} = \phi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{B \times H \times d_k \times d_v}, \quad \mathbf{Z}_{\text{sum}} = \phi(\mathbf{K})^\top \mathbf{1}_N \in \mathbb{R}^{B \times H \times d_k \times 1}
\end{equation}
\begin{equation}
\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \frac{\phi(\mathbf{Q}) \mathbf{S}_{KV}}{\phi(\mathbf{Q}) \mathbf{Z}_{\text{sum}} + \epsilon} \in \mathbb{R}^{B \times H \times N \times d_v}
\end{equation}
خروجی هدها ادغام شده و با پروجکشن خروجی فاکتورگیری‌شده بازتولید می‌گردد.""",
    r"""این ساختار فشرده‌سازی دو سطحی را محقق می‌سازد: پارامترهای لایه‌های پروجکشن با ضریب $D / (2r)$ کاهش یافته و پیچیدگی زمانی از $\mathcal{O}(N^2 d)$ به $\mathcal{O}(N \cdot d_k \cdot d_v)$ کاملاً خطی می‌شود."""
)

header = r"""% ==============================================================================
% بخش سوم: تحلیل جامع معماری و روابط ریاضی مکانیزم‌های توجه (ماژول‌های ۵۷ الی ۸۴)
% فایل: attentions_latex/part3.tex
% ==============================================================================
"""

full_content = header + "\n\n" + "\n\n".join(modules)

output_file = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/part3.tex"
with open(output_file, "w", encoding="utf-8") as fp:
    fp.write(full_content)

print(f"Successfully generated {output_file} with {len(modules)} modules.")
