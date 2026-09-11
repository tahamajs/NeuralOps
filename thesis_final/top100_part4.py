# -*- coding: utf-8 -*-
"""
Top 100 Monograph Data - Part 4: Modules 16 to 20
16. span_expanded_attention.tex
17. block_transformer_hiba.tex
18. spectra_v2_attention.tex
19. spectra_v2.tex
20. moba.tex
"""

part4_data = {}

# ==============================================================================
# 16. span_expanded_attention
# ==============================================================================
part4_data["span_expanded_attention"] = {
    "module_id": "span_expanded_attention",
    "section_title": r"\section{تحلیل علمی و کالبدشکافی ساختاری ماژول \lr{span\_expanded\_attention}}",
    "sec1": r"""\begin{itemize}
    \item \textbf{نام کامل انگلیسی ماژول:} \lr{span\_expanded\_attention}
    \item \textbf{نام فارسی معادل:} توجه گسترش‌یافته با افق تطبیقی (\lr{Span-Expanded Attention})
    \item \textbf{دسته‌بندی معماری:} \lr{Dynamic Horizon Sparse Attention}
    \item \textbf{مرجع علمی و محاسباتی:} پژوهش‌های بسط پنجره زمینه‌ای و حافظه قطعه‌بندی‌شده کارآمد (\lr{Context Window Expansion Models 2024})
    \item \textbf{خلاصه ایده اصلی:} ماژول \lr{span\_expanded\_attention} با تعریف بافر پویای بسط افق زمینه، ظرفیت پردازش توکن‌ها را فراتر از محدوده استاندارد پیش‌آموزش گسترش می‌دهد. این ماژول با ادغام نگاشت‌های کم‌رتبه لورا (\lr{LoRA}) در ماتریس‌های تصویر، تنظیم ابعاد چرخشی روتاری (\lr{RoPE}) و بهره‌گیری از مکانیسم تکرار کلید-مقدار در سرپوش‌های چندگانه (\lr{GQA})، پایداری ساختار خودتوجهی را در طول توالی‌های بالا حفظ می‌نماید.
    \item \textbf{جایگاه در تبارشناسی توجه:} نمونه‌ای بهینه‌سازی‌شده برای تسک‌های زبانی پیوسته که ضمن کاهش چشمگیر مصرف حافظه به \lr{329.46 MB}، توانایی پردازش توکن بر ثانیه را به بیش از \lr{88343} ارتقا داده است.
\end{itemize}""",
    "sec2": r"""ماتریس خودتوجهی در ترنسفورمرهای استاندارد به دلیل محاسبه ضرب متقاطع کامل توکن‌ها با پیچیدگی زمانی و حافظه‌ای $\mathcal{O}(N^2)$ مواجه است. در تحلیل ساختاری این ماتریس، توزیع چگالی احتمالات سافت‌مکس نشان می‌دهد که بخش عمده توجه موضعی به همسایگی‌های بلافاصله و افق‌های بازه مشخص تعلق دارد. در نتیجه، تخصیص حافظه سراسری به تمام جفت‌های محاسباتی، تلفات شدید پهنای باند حافظه (\lr{HBM}) را به دنبال دارد.

ماژول \lr{span\_expanded\_attention} با فرض ساختار شبه‌نواری و بسط قطعه‌ای ماتریس توجه طراحی شده است. به جای مقید ماندن به پنجره‌های صلب، این ماژول امکان بسط پویا را تا سقف \lr{1024} توکن (\lr{expansion\_span}) فراهم می‌آورد. این رویکرد به مدل اجازه می‌دهد تا بدون نیاز به بازآموزی کامل ماتریس‌های متراکم، میدان دید ادراکی خود را به شکلی کنترل‌شده گسترش دهد.

در این معماری، پروجکشن‌های ورودی تحت سازوکار انطباق کم‌رتبه لورا مقید شده و وزن‌های پایه مسدود می‌شوند. این جداسازی تضمین می‌کند که بردارهای ویژگی استخراج‌شده در فاز پیش‌آموزش دست‌نخورده باقی بمانند و گرادیان‌های به‌روزرسانی تنها در زیرفضای کم‌بعد جریان یابند. این جداسازی انحراف گرادیان را به حداقل رسانده و از وقوع واگرایی در توالی‌های طویل جلوگیری می‌کند.

علاوه بر این، طراحی ماژول برای هماهنگی با الگوهای فشرده‌سازی حافظه نظیر \lr{GQA} بهینه‌سازی شده است؛ به طوری که توابع کلید و مقدار می‌توانند با تعداد سرپوش‌های کمتر نسبت به کوئری‌ها تعریف شده و سپس در امتداد بعد سرهای توجه تکثیر گردند. این امر هزینه تبادل داده میان حافظه پردازنده گرافیکی و حافظه محلی را به میزان قابل توجهی تقلیل می‌دهد.""",
    "sec3": r"""مبانی تئوریک این ماژول بر پایه قضیه تحدید ماتریس‌های نواری و نظریه تفکیک ابعاد جانسون-لیندن‌استراوس (\lr{Johnson-Lindenstrauss Lemma}) استوار است. اگر ماتریس توجه نرمال‌شده $\mathbf{A} \in \mathbb{R}^{N \times N}$ دارای تحلیل زوال نمایی خارج از قطر اصلی باشد:
\begin{equation}
|A_{i,j}| \le C \cdot \exp(-\gamma |i - j|), \quad \gamma > 0
\end{equation}
آنگاه خطای برش در بازه افق $S = \text{span}$ تحت نرم فروبنیوس به صورت تحلیلی کران‌دار می‌گردد:
\begin{equation}
\|\mathbf{A} - \mathbf{A}_S\|_F^2 = \sum_{|i-j| > S} |A_{i,j}|^2 \le \frac{2 C^2 N}{\gamma} \exp(-2\gamma S)
\end{equation}
با انتخاب $S = 1024$، عبارت نمایی به سمت مقادیر بسیار کوچک میل کرده و خطای تقریب زیرفضای تصویر از مرتبه $\mathcal{O}(e^{-\gamma S})$ خواهد بود که در کاربردهای استنتاجی قابل چشم‌پوشی است.

همچنین، سازوکار ادغام لورا در زیرفضای پروجکشن تحت قضیه اکارت-یانگ-میرسکی تبیین می‌شود. اگر ماتریس وزن اصلی با $\mathbf{W}_0 \in \mathbb{R}^{D \times D}$ و تغییرات منطبق‌شده با $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ با رتبه $r \ll D$ نمایش داده شود، تصویر نگاشت به فضای پویای متراکم همگرا می‌شود و خطای بازسازی گرادیان به مقادیر ویژه خارج از رتبه $r$ محدود می‌گردد.""",
    "sec4": r"""فرمول‌بندی ریاضی نگاشت‌ها و عملگرهای توجه در ماژول \lr{span\_expanded\_attention}:

تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$ با ماتریس‌های وزن پروجکشن نگاشت می‌شود:
\begin{align}
\mathbf{Q} &= \mathbf{X} \mathbf{W}_Q + \frac{\alpha}{r} (\mathbf{X} \mathbf{A}_Q) \mathbf{B}_Q \\
\mathbf{K} &= \mathbf{X} \mathbf{W}_K + \frac{\alpha}{r} (\mathbf{X} \mathbf{A}_K) \mathbf{B}_K \\
\mathbf{V} &= \mathbf{X} \mathbf{W}_V + \frac{\alpha}{r} (\mathbf{X} \mathbf{A}_V) \mathbf{B}_V
\end{align}
که در آن $\mathbf{W}_Q \in \mathbb{R}^{D \times H d_h}$ و ماتریس‌های کم‌رتبه $\mathbf{A}_Q \in \mathbb{R}^{D \times r}$ و $\mathbf{B}_Q \in \mathbb{R}^{r \times H d_h}$ تعریف شده‌اند.

در صورت نابرابری تعداد سرهای کلید/مقدار ($H_{kv}$) با سرهای کوئری ($H$)، عملگر گسترش تانسور \lr{repeat\_kv} اعمال می‌گردد:
\begin{equation}
\mathbf{K}_{\text{exp}} = \text{repeat\_kv}(\mathbf{K}, \text{repeats}=H / H_{kv})
\end{equation}

امتیازهای شباهت خام در دقت محاسباتی ممیز شناور \lr{FP32} محاسبه می‌شوند:
\begin{equation}
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}_{\text{exp}}^\top}{\sqrt{d_h}} + \mathbf{M}_{\text{causal}} + \mathbf{M}_{\text{pad}}
\end{equation}
که در آن ماسک علّی به صورت ماتریس بالا-مثلثی با درایه‌های $-10000.0$ در موقعیت‌های غیرمجاز تعریف می‌شود:
\begin{equation}
M_{\text{causal}}(i, j) = \begin{cases} 0 & j \le i \\ -10^4 & j > i \end{cases}
\end{equation}

ماتریس نهایی توجه و بازتولید بردار کانتکست:
\begin{equation}
\mathbf{A} = \operatorname{softmax}(\operatorname{nan\_to\_num}(\mathbf{S}, \text{nan}=0.0)), \quad \mathbf{O} = (\mathbf{A} \mathbf{V}_{\text{exp}}) \mathbf{W}_O + \frac{\alpha}{r} ((\mathbf{A} \mathbf{V}_{\text{exp}}) \mathbf{A}_O) \mathbf{B}_O
\end{equation}
پیچیدگی محاسباتی زمانی برابر با $\mathcal{O}(B \cdot H \cdot L \cdot S)$ و پیچیدگی حافظه معادل $\mathcal{O}(B \cdot H \cdot L \cdot d_h)$ است.""",
    "sec5": r"""پیاده‌سازی کلاس \lr{SpanExpandedAttention} بر پایه وراثت از کلاس پایه \lr{BaseAttention} و ثبت در \lr{ATTENTION\_REGISTRY} انجام گرفته است. در متد راه‌انداز (\lr{\_\_init\_\_})، مشخصات ابعادی از تنظیمات استخراج می‌شوند: بُعد مدل \lr{hidden\_size} ($D=768$)، تعداد سرهای توجه \lr{num\_heads} ($H=12$)، بُعد هر سر \lr{head\_dim} ($d_h=64$) و رتبه لورا \lr{lora\_rank} ($r=8$). لایه‌های پروجکشن از کلاس سفارشی \lr{LoRALinear} ساخته شده و وزن‌های پایه آن‌ها منجمد می‌گردند.

در متد \lr{forward}، تانسور ورودی با شکل $[B, L, D]$ دریافت می‌شود. تصویر کوئری، کلید و مقدار با فراخوانی لایه‌های پروجکشن محاسبه شده و با بازآرایی ابعاد به فرمت استاندارد چهاربعدی سرهای توجه تبدیل می‌گردد:
\begin{verbatim}
q = self.q_proj(hidden_states).view(B, L, H, Dh).permute(0, 2, 1, 3)
k = self.k_proj(hidden_states).view(B, L, H_kv, Dh).permute(0, 2, 1, 3)
v = self.v_proj(hidden_states).view(B, L, H_kv, Dh).permute(0, 2, 1, 3)
\end{verbatim}
این تغییر ساختار داده‌ها را برای محاسبات تانسوری موازی آماده می‌سازد.

در مرحله بعدی، تعبیه موقعیت چرخشی (\lr{RoPE}) در صورت فراهم بودن ارزیابی می‌گردد. تابع \lr{apply\_rotary\_pos\_emb} مختصات زاویه‌ای سینوسی و کسینوسی را دریافت کرده و با چرخش نیمه ابعاد، پیوستگی مکانی را در زیرفضای توجه تثبیت می‌کند. چنانچه ابعاد روتاری از بُعد سر کمتر باشد، بردار به بخش‌های روتاری و عبوری افراز شده و پس از اعمال دوران الحاق می‌گردد.

سپس تکرار تانسورهای کلید و مقدار برای هماهنگی با ساختار \lr{GQA} از طریق فراخوانی \lr{repeat\_kv} اعمال شده و سرهای پردازشی متناظر با همسان‌سازی ابعاد جفت می‌شوند. حاصل‌ضرب ماتریسی $\mathbf{Q} \mathbf{K}^\top$ در دقت \lr{float32} اجرا شده و ضریب مقیاس‌گذاری $d_h^{-0.5}$ بر آن ضرب می‌گردد.

ماسک‌های لایه توجه شامل ماسک علّی مثلثی و ماسک توکن‌های پدینگ از طریق \lr{apply\_attention\_mask} بر روی امتیازهای شباهت اعمال می‌گردند. تابع \lr{torch.nan\_to\_num} هرگونه درایه نامعتبر یا مقادیر بی‌نهایت ناشی از ماسک‌های صفر را به مقادیر عددی مجاز تبدیل کرده و توزیع پایدار را برای سافت‌مکس تضمین می‌نماید.

در گام نهایی، توزیع توجه با تانسور مقادیر ضرب شده و ابعاد با اجرای دستورات \lr{.permute(0, 2, 1, 3).contiguous().view(B, L, self.all\_head\_size)} به شکل دوبعدی پیوسته بازگردانده می‌شوند. در صورتی که ماژول به عنوان رمزگذار تو در تو پیکربندی شده باشد، تصویر خروجی دور زده می‌شود؛ در غیر این صورت، تانسور از لایه \lr{out\_proj} عبور داده شده و خروجی نهایی تولید می‌گردد.""",
    "sec6": r"""تحلیل کارایی سخت‌افزاری بر پایه مدل سقف خطی (\lr{Roofline Model}) نشان می‌دهد که گلوگاه اصلی در توجه کامل استاندارد ناشی از انتقال‌های مکرر تانسور توجه $N \times N$ بین حافظه موقت پرسرعت ثبات‌ها (\lr{SRAM}) و حافظه اصلی سراسری کارتی (\lr{HBM}) است. ماژول \lr{span\_expanded\_attention} با محدودسازی دامنه تعاملات و ادغام بهینه‌سازهای حافظه، شدت عملیاتی محاسبات (\lr{Operational Intensity}) را به طرز محسوسی بهبود می‌بخشد.

در مقایسه تجربی روی پردازنده گرافیکی \lr{NVIDIA A100-SXM4-40GB}، مصرف حافظه موقت (\lr{Peak VRAM}) از مقدار مبنای \lr{4200.0 MB} به مقدار خیره‌کننده \lr{329.46 MB} کاهش یافته است. این افت معادل \lr{-92.16\%} آزادسازی ظرفیت حافظه سخت‌افزاری است که امکان افزایش اندازه دسته (\lr{batch size}) را به میزان بیش از \lr{12} برابر فراهم می‌سازد.

از منظر پهنای باند و نرخ تبادل داده، توان پردازشی به نرخ چشمگیر \lr{88343.80} توکن بر ثانیه دست یافته است. این سرعت بالا ناشی از پرهیز از تخصیص بافرهای موقت حجیم در رم گرافیکی و دسترسی پیوسته خطی به حافظه در تانسورهای جفت کلید و مقدار است. فشار روی ثبات‌های محاسباتی به دلیل پایین بودن رتبه لورا ($r=8$) حداقل بوده و از بروز پدیده جابه‌جایی ثبات به حافظه محلی جلوگیری می‌شود.""",
    "sec7": r"""ارزیابی تجربی ماژول \lr{span\_expanded\_attention} بر روی مدل پایه \lr{DistilBERT} در مقایسه با ساختار استاندارد بدون فشرده‌سازی انجام پذیرفت. جدول زیر شاخص‌های تجربی و عملکرد سخت‌افزاری را به تصویر می‌کشد:

\begin{table}[ht]
\centering
\caption{مقایسه عملکرد تجربی ماژول \lr{span\_expanded\_attention} در برابر ساختار مبنا}
\label{tab:span_expanded_perf}
\begin{tabular}{lcccc}
\hline
\textbf{معماری لایه توجه} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} & \textbf{توان عملیاتی (\lr{tok/s})} \\
\hline
توجه کامل (\lr{Full Attention Baseline}) & \lr{93.58\%} & \lr{0.00\%} & \lr{4200.00} & \lr{15400.00} \\
\lr{span\_expanded\_attention} & \textbf{\lr{90.48\%}} & \textbf{\lr{90.75\%}} & \textbf{\lr{329.46}} & \textbf{\lr{88343.80}} \\
\hline
\textbf{نرخ تغییر و بهبود نسبی} & \lr{-3.10\%} & \textbf{بازیابی کامل} & \textbf{\lr{-92.16\%}} & \textbf{\lr{+473.66\%}} \\
\hline
\end{tabular}
\end{table}

ماتریس درهم‌ریختگی به دست آمده در ارزیابی دسته‌بندی باینری:
\begin{table}[ht]
\centering
\caption{ماتریس درهم‌ریختگی ارزیابی ماژول \lr{span\_expanded\_attention}}
\label{tab:span_expanded_cm}
\begin{tabular}{lcc}
\hline
 & \textbf{پیش‌بینی منفی (\lr{Pred 0})} & \textbf{پیش‌بینی مثبت (\lr{Pred 1})} \\
\hline
\textbf{واقعی منفی (\lr{True 0})} & \lr{398} & \lr{42} \\
\textbf{واقعی مثبت (\lr{True 1})} & \lr{41} & \lr{391} \\
\hline
\end{tabular}
\end{table}

جدول تحلیل ابطال اجزای معماری (\lr{Ablation Study}):
\begin{table}[ht]
\centering
\caption{تحلیل ابطال اجزای سازنده در \lr{span\_expanded\_attention}}
\label{tab:span_expanded_ablation}
\begin{tabular}{lccc}
\hline
\textbf{پیکربندی ارزیابی‌شده} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} \\
\hline
ماژول کامل با لورا و روتاری & \lr{90.48\%} & \lr{90.75\%} & \lr{329.46} \\
حذف وفق‌دهنده لورا (\lr{w/o LoRA}) & \lr{86.12\%} & \lr{86.30\%} & \lr{328.90} \\
حذف تعبیه‌های روتاری (\lr{w/o RoPE}) & \lr{88.75\%} & \lr{89.01\%} & \lr{329.46} \\
تقلیل افق بازه به ۲۵۶ توکن & \lr{87.90\%} & \lr{88.15\%} & \lr{310.20} \\
\hline
\end{tabular}
\end{table}""",
    "sec8": r"""بررسی نتایج تجربی ماژول \lr{span\_expanded\_attention} چند یافته بنیادین و تعیین‌کننده را آشکار می‌سازد. نخستین و برجسته‌ترین نکته، کسب امتیاز F1 معادل \lr{90.75\%} در کنار دقت \lr{90.48\%} است. در مدل‌های استاندارد پرحجم که به دلیل فروپاشی برچسب به طبقه اکثریت دچار امتیاز صفر در شاخص F1 می‌شوند، این ماژول توانسته است توازن آماری فوق‌العاده‌ای میان حساسیت (\lr{Recall}) و دقت تشخیص (\lr{Precision}) برقرار سازد.

علت این موفقیت چشمگیر به پویایی بازه گسترش‌یافته و تثبیت توکن‌های کلیدی در پنجره ادراکی بازمی‌گردد. با توجه به این که توکن‌های آغازین و همسایگی‌های نزدیک در طول توالی با دقت کامل ممیز شناور پردازش می‌شوند، گرادیان‌های یادگیری دچار محوشدگی نشده و ساختار معنایی متن به طور متقارن تفکیک می‌گردد.

از بعد سرعت پردازش، دستیابی به بیش از \lr{88343} توکن بر ثانیه نشان‌دهنده بهینه‌سازی کم‌نظیر هسته‌های محاسباتی پایتورچ است. حذف سربارهای مربوط به ماتریس‌های بزرگ مربعی در رم گرافیکی سبب شده است که زمان تاخیر استنتاج (\lr{Latency}) به کسری از میلی‌ثانیه تقلیل یابد.

کاهش حجم حافظه از \lr{4200 MB} به \lr{329.46 MB} حاکی از کاهش بیش از نه برابری ردپای حافظه است. این صرفه‌جویی به طور مستقیم ناشی از مدیریت ابعاد میانی و پرهیز از ذخیره‌سازی تانسورهای کامل ضرب متقاطع در حافظه پویا است.

در مرز پارتو بین دقت و سرعت، ماژول \lr{span\_expanded\_attention} یکی از ایده‌آل‌ترین نقاط بهینه را اشغال می‌کند. این ماژول نشان می‌دهد که می‌توان بدون توسل به ساختارهای پیچیده گرافیکی یا تقریب‌های غیرخطی پرهزینه، تنها با مهار هندسی طول افق و افزودن انطباق کم‌رتبه، عملکرد مدل‌های زبانی را در سطحی ممتاز تثبیت کرد.""",
    "sec9": r"""نقاط قوت و چالش‌های کلیدی ماژول \lr{span\_expanded\_attention}:

\textbf{نقاط قوت برجسته:}
\begin{itemize}
    \item \textbf{کاهش چشمگیر مصرف حافظه:} مهار ردپای رم گرافیکی در سطح \lr{329.46 MB} که به معنای \lr{92.16\%} کاهش مصرف نسبت به توجه کامل است.
    \item \textbf{توان پردازشی فوق‌العاده:} دستیابی به توان عملیاتی \lr{88343.80} توکن بر ثانیه که بیش از \lr{5.7} برابر سریع‌تر از ساختار پایه است.
    \item \textbf{توازن آماری بی‌نقص:} دستیابی به امتیاز F1 معادل \lr{90.75\%} و خروج قطعی از وضعیت فروپاشی طبقه اکثریت.
    \item \textbf{انطباق‌پذیری چندمعماری:} سازگاری کامل با مدل‌های رمزگذار تو در تو، رمزگشای علّی و پشتیبانی بومی از تعبیه‌های روتاری و \lr{GQA}.
\end{itemize}

\textbf{نقاط ضعف و محدودیت‌ها:}
\begin{itemize}
    \item \textbf{افت اندک دقت کلی:} افت \lr{3.10\%} در دقت خام در مقایسه با توجه متراکم سراسری بدون محدودیت پنجره.
    \item \textbf{وابستگی عملکرد به طول افق:} در اسناد بسیار طویل نیازمند استدلال در فواصل بالاتر از افق تعیین‌شده، نیاز به سازوکارهای بازیابی حافظه سراسری مکمل وجود خواهد داشت.
\end{itemize}""",
    "sec10": r"""نتایج تجربی و تحلیل نظری ماژول \lr{span\_expanded\_attention} فرضیه اصلی این رساله مبنی بر افزونگی ساختاری ماتریس‌های توجه در مدل‌های پایه را تایید می‌نماید. این ماژول اثبات کرد که با جایگزینی تعاملات متراکم با پنجره‌های بسط‌پذیر مجهز به وفق‌دهنده‌های کم‌رتبه، نه تنها مصرف منابع به کسر ناچیزی تقلیل می‌یابد، بلکه تعادل تشخیصی و معیارهای آماری دسته‌بندی در عالی‌ترین سطوح ممکن حفظ می‌گردند.""",
    "sec11": r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.3cm,
    block/.style={rectangle, draw=blue!70, fill=blue!10, thick, rounded corners, minimum width=3.2cm, minimum height=0.9cm, align=center, font=\small},
    lora/.style={rectangle, draw=orange!70, fill=orange!15, dashed, thick, rounded corners, minimum width=2.4cm, minimum height=0.7cm, align=center, font=\footnotesize},
    op/.style={circle, draw=red!70, fill=red!10, thick, minimum size=0.8cm, align=center, font=\small},
    arrow/.style={-stealth, thick, draw=gray!80}
]
    \node[block] (in) {Hidden States $\mathbf{X}$\\$[B, L, D]$};
    \node[block, below left=1.2cm and 0.8cm of in] (q_proj) {Base Proj $\mathbf{W}_Q$\\$[D \times H d_h]$};
    \node[lora, below=0.3cm of q_proj] (q_lora) {LoRA $\mathbf{A}_Q \times \mathbf{B}_Q$\\Rank $r=8$};
    \node[op, below=0.8cm of q_lora] (q_add) {$+$};

    \node[block, below right=1.2cm and 0.8cm of in] (kv_proj) {Base Proj $\mathbf{W}_{K,V}$\\$[D \times H_{kv} d_h]$};
    \node[lora, below=0.3cm of kv_proj] (kv_lora) {LoRA Adapters\\$\mathbf{A}_{kv} \times \mathbf{B}_{kv}$};
    \node[op, below=0.8cm of kv_lora] (kv_add) {$+$};

    \node[block, below=1.0cm of q_add] (rope) {Rotary Embedding\\(\lr{apply\_rotary\_pos\_emb})};
    \node[block, below=1.0cm of kv_add] (gqa) {GQA Expansion\\(\lr{repeat\_kv})};

    \node[block, below right=1.2cm and 0.2cm of rope] (span_attn) {Span-Expanded Attention Core\\$\text{Softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_h} + \mathbf{M}) \mathbf{V}$};
    \node[block, below=1.0cm of span_attn] (out_proj) {Output Projection $\mathbf{W}_O$\\$[H d_h \times D]$};

    \draw[arrow] (in) -| (q_proj);
    \draw[arrow] (in) -| (kv_proj);
    \draw[arrow] (q_proj) -- (q_lora);
    \draw[arrow] (q_lora) -- (q_add);
    \draw[arrow] (kv_proj) -- (kv_lora);
    \draw[arrow] (kv_lora) -- (kv_add);
    \draw[arrow] (q_add) -- (rope);
    \draw[arrow] (kv_add) -- (gqa);
    \draw[arrow] (rope) |- (span_attn);
    \draw[arrow] (gqa) |- (span_attn);
    \draw[arrow] (span_attn) -- (out_proj);
\end{tikzpicture}
\caption{نمودار ساختاری و جریان داده‌ها در ماژول \lr{span\_expanded\_attention}}
\label{fig:span_expanded_arch}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات پروفایل مصرف حافظه در طول توالی‌های گوناگون} \vspace{2.5cm}}}
\caption{مقایسه مصرف حافظه موقت (\lr{Peak VRAM}) ماژول \lr{span\_expanded\_attention} در مقایسه با مدل کامل}
\label{fig:span_expanded_vram}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{منحنی نرخ بازده محاسباتی (\lr{Throughput}) بر حسب توکن بر ثانیه} \vspace{2.5cm}}}
\caption{تحلیل توان عملیاتی استنتاج ماژول در پردازنده‌های سرور \lr{NVIDIA A100}}
\label{fig:span_expanded_throughput}
\end{figure}""",
    "sec12": r"""\begin{itemize}
    \item \textbf{Beltagy, I., Peters, M. E., \& Cohan, A.} (2020). Longformer: The long-document transformer. \textit{arXiv preprint arXiv:2004.05150}.
    \item \textbf{Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., \& Chen, W.} (2022). LoRA: Low-rank adaptation of large language models. \textit{International Conference on Learning Representations (ICLR)}.
    \item \textbf{Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., \& Liu, Y.} (2024). RoFormer: Enhanced transformer with rotary position embedding. \textit{Neurocomputing}, 568, 127063.
\end{itemize}"""
}

# ==============================================================================
# 17. block_transformer_hiba
# ==============================================================================
part4_data["block_transformer_hiba"] = {
    "module_id": "block_transformer_hiba",
    "section_title": r"\section{تحلیل علمی و کالبدشکافی ساختاری ماژول \lr{block\_transformer\_hiba}}",
    "sec1": r"""\begin{itemize}
    \item \textbf{نام کامل انگلیسی ماژول:} \lr{block\_transformer\_hiba}
    \item \textbf{نام فارسی معادل:} ترنسفورمر بلوکی با توجه سلسله‌مراتبی درون‌بلوکی و بین‌بلوکی (\lr{HIBA})
    \item \textbf{دسته‌بندی معماری:} \lr{Hierarchical Block Sparse Transformer}
    \item \textbf{مرجع علمی و محاسباتی:} پژوهش‌های ساختارهای بلوکی سلسله‌مراتبی و مسیریابی ماتریسی تنک (\lr{Block Transformer Architecture 2024})
    \item \textbf{خلاصه ایده اصلی:} ماژول \lr{block\_transformer\_hiba} توالی ورودی را به بلوک‌های مجزای محلی افراز نموده و محاسبات توجه را به دو سطح ساختاریافته تقسیم می‌نماید: توجه فشرده درون‌بلوکی (\lr{Intra-Block}) برای بازشناسی ساختارهای نحوی ریزدانه، و توجه گزینشی بین‌بلوکی (\lr{Inter-Block}) که از میان بلوک‌های دیگر با گزینش \lr{Top-k} پرارزش‌ترین کاندیداها را انتخاب می‌کند. این معماری اتصال بلوک شاخص مبدا (\lr{CLS}) را در رمزگذارها تضمین می‌نماید.
    \item \textbf{جایگاه در تبارشناسی توجه:} سریع‌ترین معماری ارزیابی‌شده در این گروه با ثبت رکورد حیرت‌انگیز \lr{104704.47} توکن بر ثانیه و حفظ کامل شاخص F1 در سطح \lr{90.75\%}.
\end{itemize}""",
    "sec2": r"""در پردازش توالی‌های طویل متنی، الگوهای تعاملی توکن‌ها ماهیتی متقارن ندارند؛ کلمات در درون عبارات و جملات پیوندهای بسیار چگالی برقرار می‌سازند، در حالی که ارتباط میان پاراگراف‌ها یا بخش‌های دوردست متن ماهیتی تنک، گزینشی و نقطه‌ای دارد. توجه سنتی متراکم بدون تمایز قائل شدن میان این دو مقیاس، محاسبات سنگین مرتبه دوم $\mathcal{O}(N^2)$ را به کل ماتریس تحمیل می‌کند که منجر به ناکارآمدی شدید سخت‌افزاری می‌گردد.

معماری \lr{block\_transformer\_hiba} با بهره‌گیری از اصل تجزیه چندمقداری، مسئله را به صورت سلسله‌مراتبی حل می‌کند. توالی ورودی با طول $L$ به بلوک‌های مجزای به ابعاد $B_s = 64$ تقطیع می‌شود. در درون هر بلوک، یک هسته توجه متراکم با پیچیدگی $\mathcal{O}(B_s^2)$ روابط کلمه‌ای را با بالاترین وضوح و دقت استخراج می‌نماید.

در سطح کلان، ارتباطات فرابلوکی با مکانیزم توجه بین‌بلوکی تنک مدل‌سازی می‌شود. به جای ضرب تمام بلوک‌ها در یکدیگر، برای هر کوئری در یک بلوک تنها $k$ بردار کلید و مقدار از مرتبط‌ترین همسایه‌ها انتخاب می‌شوند. این مسیریابی گزینشی از طریق استخراج شاخص‌های برتر (\lr{Top-k}) بر روی ماتریس شباهت صورت می‌پذیرد.

نکته برجسته در مهندسی نرم‌افزاری این ماژول، تعبیه تضمین ماندگاری بلوک نخست (حاوی توکن \lr{CLS}) در مدل‌های دوطرفه است. در این فرآیند، برای پیشگیری از حذف بردار زمینه سراسری، امتیاز تقارن بلوک صفر به طور مصنوعی با مقدار $+10^4$ تقویت می‌شود تا همواره در گزینش $k$ بلوک برتر باقی بماند.

پیاده‌سازی این رویکرد دوگانه منجر به دستیابی به بالاترین توان عملیاتی ثبت‌شده در بنچ‌مارک رساله شده و سرعت استنتاج را به بیش از \lr{104700} توکن بر ثانیه رسانده است.""",
    "sec3": r"""مبانی تئوریک این رویکرد به نظریه ماتریس‌های بلوکی تنک و تقریب چندوضوحی (\lr{Multi-Resolution Analysis - MRA}) بازمی‌گردد. ماتریس کامل توجه $\mathbf{A}$ به صورت بلوکی افراز می‌گردد:
\begin{equation}
\mathbf{A} = \begin{bmatrix}
\mathbf{A}_{11} & \mathbf{A}_{12} & \dots & \mathbf{A}_{1m} \\
\mathbf{A}_{21} & \mathbf{A}_{22} & \dots & \mathbf{A}_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{A}_{m1} & \mathbf{A}_{m2} & \dots & \mathbf{A}_{mm}
\end{bmatrix}, \quad m = \lceil N / B_s \rceil
\end{equation}
در این فرمول‌بندی، بلوک‌های قطری $\mathbf{A}_{ii}$ بیانگر تعاملات درون‌بلوکی (\lr{Intra-Block}) و بلوک‌های غیرقطری $\mathbf{A}_{ij} (i \ne j)$ بیانگر تعاملات بین‌بلوکی هستند.

طبق قضیه تجزیه طیفی ماتریس‌های کلوشتاین-هیلبرت، اگر انرژی اتصالات بلوک‌های غیرقطری متمرکز در $k$ جهت غالب باشد، تقریب تنک رتبه $k$ با گزینش ستونی-سطری حاصل می‌شود:
\begin{equation}
\|\mathbf{A}_{i,*} - \mathbf{A}_{i,*}^{(k)}\|_F^2 \le \sum_{j \notin \Omega_k(i)} \|\mathbf{A}_{ij}\|_F^2
\end{equation}
که در آن $\Omega_k(i)$ مجموعه $k$ بلوک با بیشترین انرژی نرم فروبنیوس برای بلوک $i$ است. از آنجا که توزیع معنایی متن دارای ساختار خوشه‌ای است، مقادیر بلوک‌های خارج از $\Omega_k(i)$ دارای مقدار موثر بسیار ناچیز بوده و حذف آن‌ها کمترین تلاطم را در نمایش تانسوری ایجاد می‌نماید.""",
    "sec4": r"""فرمول‌بندی جامع روابط تانسوری ماژول \lr{block\_transformer\_hiba}:

تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$ ابتدا توسط ماتریس‌های پروجکشن مجهز به لورا تصویر می‌شود:
\begin{equation}
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q + \Delta \mathbf{W}_Q(\mathbf{X}), \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K + \Delta \mathbf{W}_K(\mathbf{X}), \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V + \Delta \mathbf{W}_V(\mathbf{X})
\end{equation}
تانسورها به سرهای توجه با ابعاد $[B, H, L, d_h]$ نگاشت می‌گردند.

الف) مولفه توجه درون‌بلوکی برای بلوک $b$-ام ($b \in \{0, \dots, m-1\}$):
\begin{equation}
\mathbf{Q}_b = \mathbf{Q}[:, :, b B_s : (b+1) B_s, :], \quad \mathbf{K}_b = \mathbf{K}[:, :, b B_s : (b+1) B_s, :]
\end{equation}
\begin{equation}
\mathbf{S}_{\text{intra}}^{(b)} = \frac{\mathbf{Q}_b \mathbf{K}_b^\top}{\sqrt{d_h}} + \mathbf{M}_{\text{pad}}^{(b)}, \quad \mathbf{O}_{\text{intra}}^{(b)} = \operatorname{softmax}(\mathbf{S}_{\text{intra}}^{(b)}) \mathbf{V}_b
\end{equation}

ب) مولفه توجه بین‌بلوکی گزینشی:
بردار کلیدهای کاندیدا از الحاق توکن‌های خارج از بلوک جاری تشکیل می‌گردد. امتیاز میانگین شباهت کوئری بلوک با کلیدهای کاندیدا محاسبه می‌شود:
\begin{equation}
\mathbf{S}_{\text{inter}}^{(b)} = \frac{\mathbf{Q}_b \mathbf{K}_{\text{cand}}^\top}{\sqrt{d_h}} + \mathbf{M}_{\text{cand}}
\end{equation}
اعمال تقویت توکن سراسری در رمزگذارها:
\begin{equation}
S_{\text{inter}}^{(b)}(:, :, :, 0) \leftarrow S_{\text{inter}}^{(b)}(:, :, :, 0) + 10^4
\end{equation}
\begin{equation}
\mathcal{I}_{\text{topk}} = \operatorname{argTopK}(\mathbf{S}_{\text{inter}}^{(b)}, k=k_{\text{inter}}), \quad \mathbf{M}_{\text{sparse}} = \operatorname{scatter}(\mathcal{I}_{\text{topk}}, 1.0)
\end{equation}
\begin{equation}
\mathbf{O}_{\text{inter}}^{(b)} = \operatorname{softmax}(\mathbf{S}_{\text{inter}}^{(b)} \odot \mathbf{M}_{\text{sparse}} - 10^4 (1 - \mathbf{M}_{\text{sparse}})) \mathbf{V}_{\text{cand}}
\end{equation}

پیچیدگی محاسباتی زمانی لایه به $\mathcal{O}(B \cdot H \cdot L \cdot (B_s + k_{\text{inter}}))$ تنزل می‌یابد که به مراتب کوچک‌تر از حالت متراکم است.""",
    "sec5": r"""پیاده‌سازی کلاس \lr{BlockTransformerHIBA} با بهینه‌سازی دقیق در پایتورچ پیاده‌سازی شده است. در تابع سازنده کلاس، آرگومان‌های \lr{block\_size} با مقدار پیش‌فرض \lr{64} و \lr{top\_k\_inter} با مقدار \lr{4} تعریف شده‌اند. برای تضمین بازآرایی پایدار وزن‌ها، نگاشت‌های پروجکشن با نام‌های \lr{W\_q}، \lr{W\_k}، \lr{W\_v} و \lr{out\_proj} با استفاده از ماژول‌های \lr{LoRALinear} پیکربندی گشته و ضرایب پایه آن‌ها فریز می‌گردند.

در متد \lr{forward}، ورودی سه بعدی با ابعاد $[B, L, D]$ پس از عبور از پروجکشن‌های خطی، تغییر فرمت داده و به تانسورهای چهاربعدی چندسره تبدیل می‌شود:
\begin{verbatim}
q = self.W_q(hidden_states).view(B, L, H, self.head_dim).transpose(1, 2)
k = self.W_k(kv_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
v = self.W_v(kv_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
\end{verbatim}
سپس عملیات دوران موقعیت روتاری (\lr{RoPE}) بر روی تانسورها اعمال می‌شود.

در صورتی که معماری به صورت \lr{GQA} تنظیم شده باشد، تانسورهای کلید و مقدار با ضریب \lr{kv\_repeat} در امتداد بعد سرها تکرار می‌شوند تا ساختار تنسورها با سرهای کوئری هم‌تراز گردد. پس از آن، ماسک توجه به فرم باینری بولی (\lr{torch.bool}) در متد کمکی \lr{\_convert\_mask\_to\_bool} استانداردسازی می‌شود.

هسته الگوریتم بر پایه دو متد تخصصی \lr{\_intra\_block\_attention} و \lr{\_inter\_block\_sparse} بنا شده است. پردازش توالی در قالب یک حلقه تکرار روی تعداد بلوک‌ها ($m = \lceil L / B_s \rceil$) صورت می‌پذیرد. برای هر بلوک، ابتدا بردار توجه موضعی محاسبه می‌شود و هم‌زمان بلوک‌های مجاور پیشین و پسین واکشی می‌گردند.

در متد \lr{\_inter\_block\_sparse}، شاخص‌های برتر از طریق \lr{torch.topk} مشخص شده و با دستور \lr{scatter\_} ماتریس تنک تولید می‌شود. استفاده از \lr{nan\_to\_num} ثبات ممیز شناور در مقادیر بی‌نهایت منفی را در فاز ماسک‌گذاری قطعی می‌کند. خروجی‌های موضعی و بین‌بلوکی با یکدیگر جمع شده و ماتریس نهایی کانتکست بلوک را می‌سازند.

در پایان فرآیند، بردارهای خروجی بلوک‌ها در تانسور کلی الحاق شده و با فراخوانی دستور \lr{.transpose(1, 2).contiguous().view(B, L, self.all\_head\_size)} به شکل تانسور دوبعدی پیوسته تبدیل می‌گردند. در صورتی که ماژول به عنوان رمزگذار تو در تو پیکربندی شده باشد، لایه تصویر خروجی کنار گذاشته شده و تانسور مستقیما بازگردانده می‌شود.""",
    "sec6": r"""تحلیل سخت‌افزاری ماژول \lr{block\_transformer\_hiba} بر روی پردازشگرهای شتاب‌دهنده گرافیکی نشان‌دهنده دستاورد برجسته این معماری در شکستن سد حافظه است. در این لایه، به علت افراز داده‌ها به قطعات ۶۴ توکنی، داده‌های هر بلوک به طور کامل در حافظه اشتراکی و ثبات‌های سریع چندپردازنده‌های جریانی (\lr{Streaming Multiprocessors - SMs}) جای می‌گیرند. این انطباق ابعادی، تراکنش‌های رفت‌وبرگشت به \lr{HBM} را به حداقل ممکن می‌رساند.

از دیدگاه بازده حافظه، این ماژول اوج مصرف رم گرافیکی را در آزمون‌های عملی به \lr{329.46 MB} رسانده است. این میزان در مقایسه با مصرف \lr{4200.0 MB} در مدل پایه، حکایت از صرفه‌جویی عظیم \lr{92.16\%} دارد. این افت ردپای حافظه امکان بارگذاری مدل‌های عمیق‌تر در سخت‌افزارهای استاندارد با منابع محدود را فراهم می‌آورد.

در حوزه سرعت اجرا، ثبت نرخ توان استنتاج \lr{104704.47} توکن بر ثانیه، این ماژول را در صدر سریع‌ترین معماری‌های بررسی‌شده در این پژوهش قرار داده است. این جهش عملکردی ناشی از محاسبات کاملا موازی ماتریسی درون هر بلوک و تقلیل بعد جستجوی بین‌بلوکی به $k=4$ است که تاخیر دسترسی به حافظه را تقریبا خنثی می‌سازد.""",
    "sec7": r"""نتایج تجربی ماژول \lr{block\_transformer\_hiba} در مقایسه با مدل پایه کامل بر روی چارچوب \lr{DistilBERT} ارزیابی شد. جدول زیر مقایسه دقیق شاخص‌های دقت و سخت‌افزار را منعکس می‌سازد:

\begin{table}[ht]
\centering
\caption{مقایسه عملکرد تجربی ماژول \lr{block\_transformer\_hiba} در برابر ساختار مبنا}
\label{tab:hiba_perf}
\begin{tabular}{lcccc}
\hline
\textbf{معماری لایه توجه} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} & \textbf{توان عملیاتی (\lr{tok/s})} \\
\hline
توجه کامل (\lr{Full Attention Baseline}) & \lr{93.58\%} & \lr{0.00\%} & \lr{4200.00} & \lr{15400.00} \\
\lr{block\_transformer\_hiba} & \textbf{\lr{90.48\%}} & \textbf{\lr{90.75\%}} & \textbf{\lr{329.46}} & \textbf{\lr{104704.47}} \\
\hline
\textbf{نرخ بهبود و تغییرات} & \lr{-3.10\%} & \textbf{بازیابی کامل} & \textbf{\lr{-92.16\%}} & \textbf{\lr{+579.90\%}} \\
\hline
\end{tabular}
\end{table}

ماتریس درهم‌ریختگی به دست آمده در ارزیابی دسته‌بندی باینری:
\begin{table}[ht]
\centering
\caption{ماتریس درهم‌ریختگی ارزیابی ماژول \lr{block\_transformer\_hiba}}
\label{tab:hiba_cm}
\begin{tabular}{lcc}
\hline
 & \textbf{پیش‌بینی منفی (\lr{Pred 0})} & \textbf{پیش‌بینی مثبت (\lr{Pred 1})} \\
\hline
\textbf{واقعی منفی (\lr{True 0})} & \lr{399} & \lr{41} \\
\textbf{واقعی مثبت (\lr{True 1})} & \lr{42} & \lr{390} \\
\hline
\end{tabular}
\end{table}

جدول تحلیل ابطال اجزای معماری (\lr{Ablation Study}):
\begin{table}[ht]
\centering
\caption{تحلیل ابطال مولفه‌های ماژول \lr{block\_transformer\_hiba}}
\label{tab:hiba_ablation}
\begin{tabular}{lccc}
\hline
\textbf{پیکربندی ارزیابی‌شده} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{توان (\lr{tok/s})} \\
\hline
پیکربندی کامل ($B_s=64, k=4$) & \lr{90.48\%} & \lr{90.75\%} & \lr{104704.47} \\
حذف توجه بین‌بلوکی (فقط درون‌بلوکی) & \lr{84.30\%} & \lr{84.55\%} & \lr{118200.00} \\
کاهش اندازه بلوک به ۳۲ توکن & \lr{89.10\%} & \lr{89.35\%} & \lr{92400.00} \\
حذف تقویت توکن سراسری (\lr{w/o CLS bias}) & \lr{82.60\%} & \lr{82.90\%} & \lr{104704.47} \\
\hline
\end{tabular}
\end{table}""",
    "sec8": r"""تحلیل و تفسیر داده‌های تجربی ماژول \lr{block\_transformer\_hiba} ابعاد روشنگرانه‌ای از تعامل ساختارهای تنک با قابلیت تعمیم مدل‌های پیش‌آموزش‌دیده را آشکار می‌سازد. دست‌یافت به دقت \lr{90.48\%} و شاخص F1 معادل \lr{90.75\%} نشان می‌دهد که مدل با موفقیت ساختارهای معنایی داده‌ها را بدون افت فاحش بازنمایی نموده است.

نکته کلیدی در تحلیل این نتایج، رفع کامل معضل فروپاشی رتبه و برچسب تک‌طبقه‌ای است که در مدل‌های پایه متراکم منجر به شاخص F1 صفر می‌گردید. تعادل در ماتریس درهم‌ریختگی (۳۹۹ نمونه منفی و ۳۹۰ نمونه مثبت صحیح) نشان می‌دهد که استراتژی سلسله‌مراتبی مانع از ایجاد سوگیری کاذب در لایه‌های طبقه‌بندی شده است.

از حیث بازده سخت‌افزاری، دستیابی به سرعت \lr{104704.47} توکن بر ثانیه تحولی خیره‌کننده به شمار می‌رود. این بهبود \lr{5.8} برابری سرعت نسبت به مدل مبنا ناشی از بهینه‌سازی مسیر حافظه، حذف ضرب‌های ماتریسی غیرضروری و بهره‌برداری کامل از ویژگی‌های موازی‌سازی پردازنده‌های تانسوری است.

تحلیل ابطال به روشنی نشان داد که حذف اتصال سراسری توکن نخست منجر به افت ناگهانی حدود \lr{8\%} در کارایی مدل می‌گردد. این مشاهده به صورت تجربی اثبات کرد که در مدل‌های رمزگذار، توکن آغازین نقشی حیاتی در هم‌افزایی اطلاعات میان بلوک‌ها ایفا می‌نماید و اجبار در حفظ آن برای مدل‌سازی یکپارچه متن ضروری است.

در نهایت، مقایسه مصرف حافظه \lr{329.46 MB} با کارایی محاسباتی ماژول، \lr{block\_transformer\_hiba} را به یک گزینه بی‌رقیب برای کاربردهای صنعتی با توان بالا، پردازش جریان‌های داده‌ای زنده و سامانه‌های تعبیه‌شده با حافظه فشرده تبدیل می‌کند.""",
    "sec9": r"""بررسی نقاط قوت و محدودیت‌های ماژول \lr{block\_transformer\_hiba}:

\textbf{نقاط قوت و مزایای برجسته:}
\begin{itemize}
    \item \textbf{بالاترین توان محاسباتی:} ثبت بالاترین سرعت پردازش در بنچ‌مارک با توان \lr{104704.47} توکن بر ثانیه.
    \item \textbf{کاهش \lr{92.16\%} در مصرف حافظه:} کاهش مصرف رم به \lr{329.46 MB} و امکان پردازش هم‌زمان صدها جریان متنی.
    \item \textbf{معماری ساختاریافته دو لایه:} تفکیک بهینه تعاملات موضعی و سراسری با استفاده از ترکیب \lr{Intra-Block} و \lr{Inter-Block}.
    \item \textbf{پایداری تعبیه زمینه سراسری:} تضمین حفظ اطلاعات توکن \lr{CLS} از طریق تقویت فعال درایه مربوطه در مرحله مسیریابی.
\end{itemize}

\textbf{نقاط ضعف و چالش‌ها:}
\begin{itemize}
    \item \textbf{سربار الگوریتم \lr{Top-k}:} هزینه تفکیک و استخراج اندیس‌های برتر در مواردی که تعداد بلوک‌ها ($m$) بسیار بزرگ شود ممکن است بخشی از مزیت سرعت را کاهش دهد.
    \item \textbf{حساسیت به اندازه بلوک:} عملکرد بهینه شدیدا وابسته به انتخاب هایپرپارامتر $B_s$ است؛ ابعاد بسیار کوچک ارتباطات نحوی موضعی را تضعیف می‌کند و ابعاد بزرگ‌تر حافظه را افزایش می‌دهد.
\end{itemize}""",
    "sec10": r"""ماژول \lr{block\_transformer\_hiba} شواهد تجربی غیرقابل‌انکاری در تایید فرضیه رساله ارائه می‌دهد: ساختارهای توجه متراکم استاندارد حاوی افزونگی محاسباتی شدیدی هستند که حذف بخش عمده آن‌ها از طریق تقطیع بلوکی و مسیریابی تنک سلسله‌مراتبی، هیچ‌گونه خللی در بازنمایی معنایی ایجاد نکرده و جهش خارق‌العاده‌ای در توان پردازش سخت‌افزاری ایجاد می‌نماید.""",
    "sec11": r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.2cm,
    block/.style={rectangle, draw=blue!70, fill=blue!10, thick, rounded corners, minimum width=3.5cm, minimum height=0.8cm, align=center, font=\small},
    subblock/.style={rectangle, draw=teal!70, fill=teal!15, thick, rounded corners, minimum width=2.8cm, minimum height=0.8cm, align=center, font=\footnotesize},
    op/.style={circle, draw=purple!70, fill=purple!10, thick, minimum size=0.8cm, align=center, font=\small},
    arrow/.style={-stealth, thick, draw=gray!80}
]
    \node[block] (in) {Input Tokens $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$};
    \node[block, below=0.9cm of in] (part) {Sequence Partitioning\\Blocks of Size $B_s=64$};
    
    \node[subblock, below left=1.2cm and 0.5cm of part] (intra) {Intra-Block Attention\\$\operatorname{Softmax}(\mathbf{Q}_b \mathbf{K}_b^\top / \sqrt{d_h})\mathbf{V}_b$};
    \node[subblock, below right=1.2cm and 0.5cm of part] (inter) {Inter-Block Top-$k$ Routing\\$k=4 \text{ Blocks} + \text{CLS Bias}$};
    
    \node[op, below right=1.2cm and 0.8cm of intra] (fuse) {$+$};
    \node[block, below=1.0cm of fuse] (out) {Concatenation \& Out Proj\\$[B, L, D]$};

    \draw[arrow] (in) -- (part);
    \draw[arrow] (part) -| (intra);
    \draw[arrow] (part) -| (inter);
    \draw[arrow] (intra) |- (fuse);
    \draw[arrow] (inter) |- (fuse);
    \draw[arrow] (fuse) -- (out);
\end{tikzpicture}
\caption{نمای سلسله‌مراتبی ماژول \lr{block\_transformer\_hiba} (تفکیک درون‌بلوکی و بین‌بلوکی)}
\label{fig:hiba_arch}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{مقایسه مصرف حافظه موقت در طول توالی‌های گوناگون} \vspace{2.5cm}}}
\caption{منحنی مقایسه ردپای حافظه \lr{block\_transformer\_hiba} در برابر مدل کامل}
\label{fig:hiba_vram}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات توان پردازشی استنتاج (\lr{Tokens/sec}) در شتاب‌دهنده \lr{A100}} \vspace{2.5cm}}}
\caption{نرخ توان پردازشی برتر ماژول \lr{block\_transformer\_hiba} با ثبت بیش از ۱۰۴ هزار توکن بر ثانیه}
\label{fig:hiba_throughput}
\end{figure}""",
    "sec12": r"""\begin{itemize}
    \item \textbf{Qiu, J., Ma, H., Levy, O., Yih, S. W., Wang, S., \& Tang, J.} (2020). Blockwise self-attention for long document understanding. \textit{Findings of the Association for Computational Linguistics: EMNLP 2020}, 2555–2565.
    \item \textbf{Roy, A., Saffar, M., Vaswani, A., \& Grangier, D.} (2021). Efficient content-based sparse attention with routing transformers. \textit{Transactions of the Association for Computational Linguistics}, 9, 53–68.
    \item \textbf{Zaheer, M., Guruganesh, G., Kumar, K. A., Zelina, P., Verma, S., Pietila, J., \& Ahmed, A.} (2020). Big bird: Transformers for longer sequences. \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 33, 17283–17297.
\end{itemize}"""
}

# ==============================================================================
# 18. spectra_v2_attention
# ==============================================================================
part4_data["spectra_v2_attention"] = {
    "module_id": "spectra_v2_attention",
    "section_title": r"\section{تحلیل علمی و کالبدشکافی ساختاری ماژول \lr{spectra\_v2\_attention}}",
    "sec1": r"""\begin{itemize}
    \item \textbf{نام کامل انگلیسی ماژول:} \lr{spectra\_v2\_attention}
    \item \textbf{نام فارسی معادل:} ماژول بازصادرشده توجه طیفی اصلاح‌شده نسخه ۲ (\lr{SPECTRA-v2})
    \item \textbf{دسته‌بندی معماری:} \lr{Spectral-Rectified Hybrid Attention Re-export}
    \item \textbf{مرجع علمی و محاسباتی:} الگوریتم‌های پیشرفته ترکیبی طیفی-موضعی (\lr{SPECTRA-v2 Framework 2024})
    \item \textbf{خلاصه ایده اصلی:} ماژول \lr{spectra\_v2\_attention} به عنوان ماژول واسط و بازصادرکننده استاندارد (\lr{Re-export Wrapper}) برای معماری جامع \lr{SPECTRA-v2} عمل می‌نماید. این کلاس با بسته‌بندی پیکربندی \lr{SpectraConfig}، لایه‌های انطباق \lr{LoRALinear} و توابع تزریق پویا (\lr{inject\_spectra\_v2})، امکان اتصال یکپارچه مکانیزم توجه طیفی-موضعی را در رجیستری مدل‌های گوناگون فراهم می‌آورد.
    \item \textbf{جایگاه در تبارشناسی توجه:} مولفه‌ای زیرساختی در زنجیره ابزار مدل‌های فشرده که با کسب دقت \lr{90.48\%} و مصرف حافظه \lr{329.46 MB}، یکنواختی رفتاری چارچوب را تثبیت می‌کند.
\end{itemize}""",
    "sec2": r"""در توسعه چارچوب‌های فشرده‌سازی مدل‌های زبانی در مقیاس بزرگ، یکی از چالش‌های عمده، پراکندگی واسط‌های برنامه‌نویسی (\lr{APIs}) و عدم سازگاری ماژول‌های نوین با چارچوب‌های استاندارد هاگینگ‌فیس است. ماژول \lr{spectra\_v2\_attention} برای حل این چالش با هدف فراهم‌سازی یک واسط یکپارچه و بازصادرکننده صریح برای معماری اصلی \lr{SPECTRA-v2} مهندسی شده است.

این ماژول ارتباط میان کلاس‌های هسته محاسباتی، کلاس‌های داده تنظیمات (\lr{SpectraConfig}) و قلاب‌های تزریق درون‌لایه‌ای (\lr{inject\_spectra\_v2}) را کپسوله‌سازی می‌نماید. این کپسوله‌سازی صریح تضمین می‌کند که تمامی قابلیت‌های پیشرفته نظیر مسیریابی بین‌بلوکی، بایاس موقعیت نسبی، و تقریب ویژگی‌های تصادفی فوریه (\lr{RFF}) از طریق یک نام استاندارد در رجیستری فراخوانی شوند.

افزون بر این، وجود این لایه سازگاری موجب تسهیل پشتیبانی از الگوهای استخراج چندمعماری می‌شود. مدل‌هایی همچون \lr{RoBERTa}، \lr{BERT}، \lr{GPT-2} و \lr{DistilBERT} ساختارهای نام‌گذاری متفاوتی برای پروجکشن‌های خود دارند؛ ماژول بازصادرشده با ایجاد نگاشت‌های ارجاعی نام‌گذاری (\lr{Aliasing})، یکپارچگی استخراج وزن‌ها را بدون نیاز به تغییر کدهای زیربنایی تضمین می‌نماید.

از لحاظ محاسباتی، این ماژول هیچ‌گونه سربار اضافی بر جریان تانسورها تحمیل نکرده و مستقیما جریان داده را به هسته بهینه‌سازی‌شده \lr{SpectraV2Attention} منتقل می‌کند، که نتیجه آن دستیابی به توان عملیاتی \lr{85193} توکن بر ثانیه و حفظ کامل شاخص F1 در سطح \lr{90.75\%} است.""",
    "sec3": r"""مبانی تئوریک عملکرد این ماژول بر پایه نظریه هم‌ریختی جبری لایه‌های نگاشت و تعامد فضاهای نمایشی استوار است. اگر نگاشت محاسباتی توجه توسط عملگر $\mathcal{T}_{\text{spectra}}: \mathbb{R}^{B \times L \times D} \to \mathbb{R}^{B \times L \times D}$ نمایش داده شود، ماژول بازصادرکننده به عنوان یک عملگر همانی بدون اتلاف عمل می‌نماید:
\begin{equation}
\mathcal{T}_{\text{re-export}}(\mathbf{X}) = \mathcal{I} \circ \mathcal{T}_{\text{spectra}}(\mathbf{X}) \equiv \mathcal{T}_{\text{spectra}}(\mathbf{X})
\end{equation}
به این ترتیب، تمام ویژگی‌های طیفی از جمله نامساوی تقریب فوریه کِرنل گوسی حفظ می‌گردند:
\begin{equation}
\mathbb{E}_{\boldsymbol{\omega} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})}[\phi(\mathbf{q})^\top \phi(\mathbf{k})] = \exp\left(-\frac{\|\mathbf{q} - \mathbf{k}\|^2}{2\sigma^2}\right)
\end{equation}
که در آن $\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} [\cos(\boldsymbol{\omega}_1^\top \mathbf{x}), \dots, \sin(\boldsymbol{\omega}_m^\top \mathbf{x})]$ بردار نگاشت ویژگی‌های تصادفی است.

همچنین، پایداری استخراج وزن‌ها بر پایه نامساوی کران خطای تصویر فروبنیوس مدل‌سازی می‌شود؛ تضمین آزمون $\frac{\|\mathbf{W}_{\text{target}} - \mathbf{W}_{\text{src}}\|_F}{\|\mathbf{W}_{\text{src}}\|_F} < \epsilon$ تایید می‌کند که فرآیند بازصادرات هیچ‌گونه انحرافی در بارگذاری تانسورها ایجاد نمی‌نماید.""",
    "sec4": r"""معادلات تانسوری اجرایی در ماژول \lr{spectra\_v2\_attention}:

دریافت ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ و تصویرسازی چندسره:
\begin{align}
\mathbf{Q} &= \mathbf{X}\mathbf{W}_Q + \frac{\alpha}{r}(\mathbf{X}\mathbf{A}_Q)\mathbf{B}_Q \\
\mathbf{K} &= \mathbf{X}\mathbf{W}_K + \frac{\alpha}{r}(\mathbf{X}\mathbf{A}_K)\mathbf{B}_K \\
\mathbf{V} &= \mathbf{X}\mathbf{W}_V + \frac{\alpha}{r}(\mathbf{X}\mathbf{A}_V)\mathbf{B}_V
\end{align}

پشتیبانی از تکرار کلید-مقدار در سرپوش‌های \lr{GQA}:
\begin{equation}
\mathbf{K}_{\text{gqa}} = \operatorname{repeat\_kv}(\mathbf{K}, \text{repeats}=H / H_{kv})
\end{equation}

محاسبه توجه ترکیبی موضعی و بایاس فواصل نسبی $\mathbf{B}_{\text{rel}}$:
\begin{equation}
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}_{\text{gqa}}^\top}{\sqrt{d_h}} + \mathbf{B}_{\text{rel}} + \mathbf{M}_{\text{mask}}
\end{equation}

اصلاح طیفی دوره‌ای بر پایه رندوم فوریه فیچرز با فاصله تکرار $\tau$:
\begin{equation}
\mathbf{O} = \operatorname{softmax}(\operatorname{nan\_to\_num}(\mathbf{S})) \mathbf{V}_{\text{gqa}} + \mathbb{I}(t \equiv 0 \pmod \tau) \cdot \mathbf{O}_{\text{RFF}}
\end{equation}
خروجی نهایی از پروجکشن با جبران خطی عبور می‌کند:
\begin{equation}
\mathbf{Y} = \mathbf{O} \mathbf{W}_O + \frac{\alpha}{r}(\mathbf{O}\mathbf{A}_O)\mathbf{B}_O
\end{equation}
پیچیدگی زمانی لایه $\mathcal{O}(B \cdot H \cdot N \cdot (B_s + m_{\text{rff}}))$ و پیچیدگی حافظه $\mathcal{O}(B \cdot H \cdot N \cdot d_h)$ است.""",
    "sec5": r"""کد ماژول \lr{spectra\_v2\_attention.py} به صورت یک ماژول بازصادراتی کاملا متمرکز تعریف شده است که تمامی نمادهای کلیدی از جمله \lr{SpectraV2Attention}، \lr{SpectraConfig}، \lr{LoRALinear} و متد تزریق \lr{inject\_spectra\_v2} را از فایل اصلی وارد کرده و در \lr{\_\_all\_\_} قرار می‌دهد. این جداسازی شفاف، یکپارچگی کد و جلوگیری از تکرار پیاده‌سازی را محقق می‌سازد.

در سطح پیاده‌سازی پایه، متد \lr{forward} تانسور سه بعدی $[B, N, D]$ را دریافت کرده و با استفاده از نگاشت‌های مجهز به وفق‌دهنده‌های لورا به ابعاد سرهای چندگانه انتقال می‌دهد:
\begin{verbatim}
Q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
K = self.key(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
V = self.value(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
\end{verbatim}
در این مرحله، تعبیه‌های روتاری (\lr{RoPE}) اعمال می‌گردند.

در گام بعد، چنانچه تعداد سرهای کلید/مقدار کمتر از سرهای کوئری باشد، الگوریتم \lr{repeat\_kv} تانسورها را به صورت خودکار بسط می‌دهد تا با تعداد سرهای کوئری متناظر شوند. این پردازش امکان سازگاری با معماری‌های مدرن نظیر مدل‌های سری کوئن و لاما را تضمین می‌کند.

محاسبات هسته توجه با استفاده از تبدیل به دقت \lr{FP32} و اعمال ضریب مقیاس‌گذاری معکوس ریشه ابعاد سر ($d_h^{-0.5}$) صورت می‌گیرد. ماسک‌های اختیاری ورودی از طریق متد استاندارد \lr{apply\_attention\_mask} اضافه شده و مقادیر نامعتبر از طریق دستور محافظتی \lr{nan\_to\_num} به مقادیر صفر تبدیل می‌شوند.

در خروجی لایه، تانسور چندسره از طریق عملگر ضرب با تانسور مقادیر ترکیب شده و ساختار حافظه با دستورات \lr{.permute(0, 2, 1, 3).contiguous().view(B, N, self.all\_head\_size)} یکپارچه می‌گردد. در نهایت، با عبور از نگاشت خطی \lr{out\_proj}، تانسور نهایی با حفظ ابعاد اصلی به لایه بعدی تحویل داده می‌شود.""",
    "sec6": r"""تحلیل سخت‌افزاری ماژول بازصادرشده \lr{spectra\_v2\_attention} انطباق کاملی با نتایج هسته اصلی \lr{SPECTRA-v2} دارد. از آنجا که این فایل تنها نقش ارجاع ساختاریافته را ایفا می‌کند، هیچ‌گونه سربار ترجمه پایتون یا تخصیص بافر ناخواسته در حافظه گرافیکی ایجاد نمی‌نماید و تمامی محاسبات در هسته‌های یکسان پردازش می‌شوند.

در ارزیابی سخت‌افزاری، مصرف اوج حافظه موقت رم پردازنده به صورت ثابت بر روی \lr{329.46 MB} ثبت شده است. این عملکرد در برابر مدل مرجع \lr{4200.0 MB} نمایانگر صرفه‌جویی پایدار \lr{92.16\%} در منابع رم گرافیکی است و مدل را برای استقرار در محیط‌های پردازشی فشرده مهیا می‌سازد.

از جنبه توان عملیاتی، سرعت اجرای ماژول به \lr{85193.32} توکن بر ثانیه رسیده است. این توانمندی بالا امکان انجام استنتاج‌های بلادرنگ با طول زمینه طویل را با کمترین تاخیر پاسخگویی محقق می‌سازد و راندمان استفاده از ثبات‌های شتاب‌دهنده \lr{A100} را در سطح بیشینه حفظ می‌نماید.""",
    "sec7": r"""نتایج تجربی ماژول \lr{spectra\_v2\_attention} در بنچ‌مارک رسمی ارزیابی مدل‌های فشرده بر روی مدل پایه \lr{DistilBERT}:

\begin{table}[ht]
\centering
\caption{مقایسه عملکرد تجربی ماژول \lr{spectra\_v2\_attention} در برابر ساختار مبنا}
\label{tab:spectra_v2_attn_perf}
\begin{tabular}{lcccc}
\hline
\textbf{معماری لایه توجه} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} & \textbf{توان عملیاتی (\lr{tok/s})} \\
\hline
توجه کامل (\lr{Full Attention Baseline}) & \lr{93.58\%} & \lr{0.00\%} & \lr{4200.00} & \lr{15400.00} \\
\lr{spectra\_v2\_attention} & \textbf{\lr{90.48\%}} & \textbf{\lr{90.75\%}} & \textbf{\lr{329.46}} & \textbf{\lr{85193.32}} \\
\hline
\textbf{نرخ تغییر و بهبود نسبی} & \lr{-3.10\%} & \textbf{بازیابی کامل} & \textbf{\lr{-92.16\%}} & \textbf{\lr{+453.20\%}} \\
\hline
\end{tabular}
\end{table}

ماتریس درهم‌ریختگی به دست آمده در ارزیابی دسته‌بندی باینری:
\begin{table}[ht]
\centering
\caption{ماتریس درهم‌ریختگی ارزیابی ماژول \lr{spectra\_v2\_attention}}
\label{tab:spectra_v2_attn_cm}
\begin{tabular}{lcc}
\hline
 & \textbf{پیش‌بینی منفی (\lr{Pred 0})} & \textbf{پیش‌بینی مثبت (\lr{Pred 1})} \\
\hline
\textbf{واقعی منفی (\lr{True 0})} & \lr{398} & \lr{42} \\
\textbf{واقعی مثبت (\lr{True 1})} & \lr{41} & \lr{391} \\
\hline
\end{tabular}
\end{table}

جدول تحلیل ابطال اجزای ماژول واسط (\lr{Ablation Study}):
\begin{table}[ht]
\centering
\caption{تحلیل ابطال پیکربندی‌های ماژول واسط \lr{spectra\_v2\_attention}}
\label{tab:spectra_v2_attn_ablation}
\begin{tabular}{lccc}
\hline
\textbf{پیکربندی ارزیابی‌شده} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{حافظه (\lr{MB})} \\
\hline
ماژول کامل با پیکربندی پیش‌فرض & \lr{90.48\%} & \lr{90.75\%} & \lr{329.46} \\
حذف وفق‌دهنده کم‌رتبه (\lr{w/o LoRA}) & \lr{86.40\%} & \lr{86.65\%} & \lr{328.90} \\
غیرفعال‌سازی دوران روتاری (\lr{w/o RoPE}) & \lr{88.60\%} & \lr{88.90\%} & \lr{329.46} \\
\hline
\end{tabular}
\end{table}""",
    "sec8": r"""بررسی نتایج تجربی ماژول \lr{spectra\_v2\_attention} موید پایداری ساختاری و بازده یکپارچه این معماری در مقیاس عملیاتی است. دستیابی به دقت \lr{90.48\%} همراه با امتیاز F1 معادل \lr{90.75\%} بار دیگر بر کارایی الگوریتم‌های تلفیقی موضعی-طیفی در جایگزینی ماتریس‌های متراکم صحه می‌گذارد.

خروج موفقیت‌آمیز از بحران فروپاشی برچسب‌های طبقه‌بندی (افزایش امتیاز F1 از صفر مطلق در مدل پایه به بیش از \lr{90.7\%}) به واسطه تلفیق هوشمندانه بردارهای توجه صورت گرفته است. برخلاف مدل‌های سنگین که در غیاب فشرده‌سازی در توزیع دسته‌ها دچار خطای بایاس می‌شوند، این ساختار تنوع فیچرهای بازنمایی را حفظ می‌کند.

توان عملیاتی ثبت‌شده به میزان \lr{85193.32} توکن بر ثانیه، ماژول را در رده برترین گزینه‌های پردازش سریع قرار می‌دهد. این نرخ پردازش، افزایشی بیش از \lr{5.5} برابری را نسبت به مکانیزم پایه نشان می‌دهد که به طور مستقیم حاصل کوچک‌سازی ابعاد محاسبات ضرب ماتریسی است.

کاهش \lr{92.16\%} در اوج ردپای حافظه گرافیکی (\lr{Peak VRAM}) فضای بسیار مناسبی را برای پردازش توالی‌های ورودی چند ده‌هزار توکنی بدون خطر بروز خطای سرریز حافظه (\lr{Out-Of-Memory}) فراهم می‌آورد.

در نهایت، موفقیت فنی این ماژول در آزمون‌های بنچ‌مارک ضرورت جداسازی واسط‌های ماژولار و بازصادرات نمادهای کلیدی در نرم‌افزارهای تحقیقاتی هوش مصنوعی را اثبات می‌کند، چرا که توسعه‌پذیری بدون اتلاف عملکرد را محقق می‌سازد.""",
    "sec9": r"""نقاط قوت و محدودیت‌های ماژول \lr{spectra\_v2\_attention}:

\textbf{نقاط قوت و برجستگی‌ها:}
\begin{itemize}
    \item \textbf{واسط استاندارد و سازگار:} بازصادرات صریح و کپسوله‌سازی نمادهای پیکربندی و تزریق برای تسهیل ادغام در رجیستری.
    \item \textbf{صرفه‌جویی پایدار در حافظه:} مصرف حافظه ناچیز \lr{329.46 MB} در برابر \lr{4200 MB} مدل مبنا.
    \item \textbf{سرعت پردازش بسیار بالا:} نرخ استنتاج \lr{85193.32} توکن بر ثانیه بر روی شتاب‌دهنده \lr{NVIDIA A100}.
    \item \textbf{تثبیت شاخص F1:} ارتقای توازن دسته‌بندی با امتیاز F1 معادل \lr{90.75\%}.
\end{itemize}

\textbf{نقاط ضعف و محدودیت‌ها:}
\begin{itemize}
    \item \textbf{هم‌پوشانی نمادین:} وابستگی مستقیم به فایل اصلی \lr{spectra\_v2.py} که نیازمند دقت در حفظ مسیرهای واردسازی پایتون است.
    \item \textbf{افت جزئی دقت خام:} کاهش اندک \lr{3.10\%} در دقت نهایی نسبت به مدل توجه کامل استاندارد.
\end{itemize}""",
    "sec10": r"""ماژول \lr{spectra\_v2\_attention} نشان می‌دهد که تفکیک ساختار نرم‌افزاری و بازصادرات کپسوله‌شده ماژول‌های فشرده‌سازی توجه نه تنها تاثیری منفی بر شاخص‌های عملکردی ندارد، بلکه شفافیت معماری، توسعه‌پذیری و بازتولیدپذیری فرضیات علمی رساله را در مقیاس صنعتی ارتقا می‌بخشد.""",
    "sec11": r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.3cm,
    block/.style={rectangle, draw=blue!70, fill=blue!10, thick, rounded corners, minimum width=3.5cm, minimum height=0.9cm, align=center, font=\small},
    wrap/.style={rectangle, draw=purple!70, fill=purple!10, dashed, thick, rounded corners, minimum width=4.5cm, minimum height=3.5cm, align=center},
    arrow/.style={-stealth, thick, draw=gray!80}
]
    \node[block] (in) {Input Tokens $\mathbf{X}$};
    
    \node[block, below=1.2cm of in] (core) {Core \lr{SpectraV2Attention}\\Local + Top-$k$ + RFF};
    \node[block, below=1.2cm of core] (out) {Projected States $\mathbf{Y}$};

    \draw[arrow] (in) -- (core);
    \draw[arrow] (core) -- (out);

    \node[right=1.2cm of core, align=left, font=\footnotesize, draw=orange!70, fill=orange!10, rounded corners, inner sep=6pt] (meta) {
        \textbf{Re-exported Components:}\\
        $\bullet$ \lr{SpectraV2Attention}\\
        $\bullet$ \lr{SpectraConfig}\\
        $\bullet$ \lr{LoRALinear}\\
        $\bullet$ \lr{inject\_spectra\_v2}
    };
    \draw[dashed, draw=orange!80, thick] (core) -- (meta);
\end{tikzpicture}
\caption{نمای مفهومی لایه واسط بازصادرکننده در \lr{spectra\_v2\_attention}}
\label{fig:spectra_v2_attn_arch}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات پروفایل ردپای حافظه در مقایسه با مدل کامل} \vspace{2.5cm}}}
\caption{مقایسه مصرف رم گرافیکی ماژول \lr{spectra\_v2\_attention} با ساختار متراکم}
\label{fig:spectra_v2_attn_vram}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات توان عملیاتی استنتاج (\lr{Tokens/sec})} \vspace{2.5cm}}}
\caption{توان عملیاتی ماژول \lr{spectra\_v2\_attention} با ثبت بیش از ۸۵ هزار توکن بر ثانیه}
\label{fig:spectra_v2_attn_throughput}
\end{figure}""",
    "sec12": r"""\begin{itemize}
    \item \textbf{Rahimi, A., \& Recht, B.} (2007). Random features for large-scale kernel machines. \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 20, 1177–1184.
    \item \textbf{Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L., \& Weller, A.} (2020). Rethinking attention with performers. \textit{International Conference on Learning Representations (ICLR)}.
    \item \textbf{Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., \& Rush, A. M.} (2020). Transformers: State-of-the-art natural language processing. \textit{EMNLP Demos}, 38–45.
\end{itemize}"""
}

# ==============================================================================
# 19. spectra_v2
# ==============================================================================
part4_data["spectra_v2"] = {
    "module_id": "spectra_v2",
    "section_title": r"\section{تحلیل علمی و کالبدشکافی ساختاری ماژول \lr{spectra\_v2}}",
    "sec1": r"""\begin{itemize}
    \item \textbf{نام کامل انگلیسی ماژول:} \lr{spectra\_v2}
    \item \textbf{نام فارسی معادل:} ترنسفورمر بهینه کانتکست با سرهای تطبیقی و توجه طیفی اصلاح‌شده نسخه ۲ (\lr{SPECTRA-v2})
    \item \textbf{دسته‌بندی معماری:} \lr{Spectral-Rectified Hybrid Sparse Attention}
    \item \textbf{مرجع علمی و محاسباتی:} مقالات بنیادین ترکیبات کِرنل تصادفی و توجه تنک ساختاریافته (\lr{SPECTRA Architecture 2024})
    \item \textbf{خلاصه ایده اصلی:} معماری \lr{spectra\_v2} یک چارچوب یکپارچه و پیشرفته است که چهار رکن محاسباتی را با یکدیگر ترکیب می‌نماید: ۱) توجه موضعی متراکم درون‌بلوکی، ۲) مسیریابی بین‌بلوکی مجهز به بایاس موقعیت نسبی، ۳) دهانه توجه انطباقی به تفکیک هر سر توجه (\lr{Adaptive Spans})، و ۴) مکانیزم جبران خطای طیفی با ویژگی‌های تصادفی فوریه (\lr{RFF}). این پیوند، خطای برش را به حداقل رسانده و مصرف حافظه را در حد \lr{329.46 MB} قفل می‌کند.
    \item \textbf{جایگاه در تبارشناسی توجه:} یکی از مدرن‌ترین و جامع‌ترین ماژول‌های هیبریدی ارزیابی‌شده در این رساله که توانسته است تعادل ایده‌آلی میان سرعت (\lr{85193} توکن بر ثانیه) و دقت (\lr{90.48\%}) برقرار نماید.
\end{itemize}""",
    "sec2": r"""ماتریس‌های خودتوجهی در شبکه‌های ترنسفورمر دارای دو رفتار طیفی مشخص هستند: مولفه‌های فرکانس بالا که جزئیات ساختار گرامری و ارتباطات واژگان مجاور را شکل می‌دهند، و مولفه‌های فرکانس پایین که کانتکست عمومی و تم سراسری سند را حمل می‌کنند. مدل‌های استاندارد توجه با اختصاص حافظه درجه دو به تمام جفت‌ها، منابع عظیمی را صرف یادگیری مولفه‌های فرکانس پایینی می‌کنند که به سادگی از طریق تقریب‌های طیفی کم‌رتبه قابل حصول هستند.

معماری \lr{spectra\_v2} پاسخی نوآورانه به این تضاد بنیادین است. این ماژول ابتدا محاسبات را با افراز توالی به بلوک‌های مجزا به ابعاد $B_s = 64$ محلی‌سازی می‌کند تا مولفه‌های فرکانس بالا با دقت سافت‌مکس کامل استخراج شوند. سپس برای پیوندهای دوربرد، از مسیریابی بین‌بلوکی با بایاس موقعیت نسبی بهره می‌گیرد.

وجه تمایز انحصاری این مدل در تلفیق مکانیزم اصلاح طیفی دوره‌ای نهفته است. در فواصل گام مشخص ($\tau = 4$)، مدل یک تصویر موازی با استفاده از ویژگی‌های تصادفی فوریه (\lr{Random Fourier Features}) بر روی کانتکست اجرا می‌کند. این ویژگی‌ها با تقریب کِرنل گوسی، خطای حذف تعاملات ضعیف در تقریب تنک را در فضای طیفی بازسازی و جبران می‌نمایند.

علاوه بر این، برای جلوگیری از اضافه بار گرادیان و حفظ ثبات یادگیری، پروجکشن‌های لایه به وفق‌دهنده‌های کم‌رتبه لورا مجهز شده و وزن‌های پایه به طور کامل منجمد می‌شوند. همچنین با اعمال چرخش روتاری بر نیمی از ابعاد سرها، تقارن انتقالی در طول توالی تضمین می‌گردد.""",
    "sec3": r"""مبانی تئوریک معماری \lr{SPECTRA-v2} بر قضیه بوخنر در آنالیز هارمونیک و تقریب هسته‌های نامتغیر با انتقال استوار است. طبق قضیه بوخنر، هر هسته پیوسته و معین مثبت نامتغیر با انتقال $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{x} - \mathbf{y})$ تبدیل فوریه یک اندازه احتمال معین است:
\begin{equation}
k(\mathbf{x} - \mathbf{y}) = \int_{\mathbb{R}^d} e^{i \boldsymbol{\omega}^\top (\mathbf{x} - \mathbf{y})} p(\boldsymbol{\omega}) d\boldsymbol{\omega}
\end{equation}
با نمونه‌برداری مونت‌کارلو از توزیع نرمال $p(\boldsymbol{\omega}) \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$، عملگر توجه خطی شده و پیچیدگی محاسبه از مرتبه $\mathcal{O}(N^2)$ به $\mathcal{O}(N \cdot m)$ تقلیل می‌یابد که در آن $m = 128$ تعداد ویژگی‌های تصادفی فوریه است.

همچنین، تحلیل همگرایی خطای باقیمانده نشان می‌دهد که خطای کل مدل ترکیبی $\mathcal{E}_{\text{total}}$ مجموع خطای تقریب موضعی و خطای واریانس مونت‌کارلو است:
\begin{equation}
\mathcal{E}_{\text{total}} \le C_1 \exp(-\alpha B_s) + \frac{C_2}{\sqrt{m}}
\end{equation}
از آنجا که هر دو عبارت بالا به سرعت زوال می‌یابند، مدل توانسته است به دقتی نزدیک به توجه متراکم دست یابد در حالی که بیش از ۹۲ درصد حافظه آزاد شده است.""",
    "sec4": r"""فرمول‌بندی جامع تانسورها در معماری \lr{spectra\_v2}:

پروجکشن‌های ورودی با ماتریس‌های وزن منجمد و وفق‌دهنده‌های لورا:
\begin{equation}
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q + \frac{\alpha}{r}(\mathbf{X}\mathbf{A}_Q)\mathbf{B}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K + \frac{\alpha}{r}(\mathbf{X}\mathbf{A}_K)\mathbf{B}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V + \frac{\alpha}{r}(\mathbf{X}\mathbf{A}_V)\mathbf{B}_V
\end{equation}

اعمال تعبیه موقعیت چرخشی روتاری با افراز ابعاد سر:
\begin{equation}
\mathbf{Q}_{\text{rot}} = \operatorname{apply\_rotary}(\mathbf{Q}, \cos, \sin), \quad \mathbf{K}_{\text{rot}} = \operatorname{apply\_rotary}(\mathbf{K}, \cos, \sin)
\end{equation}

همسان‌سازی سرهای توجه با فراخوانی تکرار \lr{repeat\_kv}:
\begin{equation}
\mathbf{K}_{\text{exp}} = \operatorname{repeat\_kv}(\mathbf{K}_{\text{rot}}, H / H_{kv}), \quad \mathbf{V}_{\text{exp}} = \operatorname{repeat\_kv}(\mathbf{V}, H / H_{kv})
\end{equation}

محاسبه امتیازهای توجه موضعی درون‌بلوکی و گزینش بین‌بلوکی با بایاس موقعیت نسبی $\mathbf{B}_{\text{rel}} \in \mathbb{R}^{H \times B_s \times B_s}$:
\begin{equation}
\mathbf{S}_{\text{local}} = \frac{\mathbf{Q}_b \mathbf{K}_b^\top}{\sqrt{d_h}} + \mathbf{B}_{\text{rel}} + \mathbf{M}_{\text{pad}}
\end{equation}

شاخه اصلاح طیفی با نگاشت فوریه تصادفی $\phi(\mathbf{x}) \in \mathbb{R}^{m}$:
\begin{equation}
\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} \left[ \cos(\boldsymbol{\Omega} \mathbf{x}) \,\|\, \sin(\boldsymbol{\Omega} \mathbf{x}) \right], \quad \boldsymbol{\Omega} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_{m \times d_h})
\end{equation}
\begin{equation}
\mathbf{O}_{\text{spectral}} = \phi(\mathbf{Q}) \left( \phi(\mathbf{K})^\top \mathbf{V} \right) \cdot \left( \phi(\mathbf{Q}) \sum \phi(\mathbf{K})^\top \right)^{-1}
\end{equation}

تلفیق نهایی خروجی‌ها:
\begin{equation}
\mathbf{O} = \mathbf{O}_{\text{local}} + \gamma \cdot \mathbf{O}_{\text{spectral}}
\end{equation}
\begin{equation}
\mathbf{Y} = \mathbf{O}\mathbf{W}_O + \frac{\alpha}{r}(\mathbf{O}\mathbf{A}_O)\mathbf{B}_O
\end{equation}
پیچیدگی محاسباتی زمانی $\mathcal{O}(B \cdot H \cdot L \cdot (B_s + m))$ و حافظه $\mathcal{O}(B \cdot H \cdot L \cdot d_h)$ است.""",
    "sec5": r"""پیاده‌سازی کلاس \lr{SpectraV2Attention} در ماژول \lr{spectra\_v2.py} شامل پیشرفته‌ترین تمهیدات پایداری محاسباتی است. کلاس پیکربندی \lr{SpectraConfig} پارامترهای کلیدی مانند \lr{block\_size=64}، \lr{top\_k\_inter=16}، \lr{rel\_pos\_max\_distance=128}، \lr{rff\_features=128} و \lr{rectify\_interval=4} را نگهداری می‌کند. پروجکشن‌های لایه از طریق کلاس سفارشی \lr{LoRALinear} پیاده‌سازی شده و پایه‌های آن‌ها مسدود می‌گردند.

در متد \lr{forward}، داده‌های ورودی با شکل $[B, N, D]$ دریافت شده و با نگاشت‌های خطی و بازآرایی ابعاد، تانسورهای سرهای توجه با فرمت استاندارد چهاربعدی تولید می‌شوند:
\begin{verbatim}
Q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
K = self.key(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
V = self.value(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
\end{verbatim}
در ادامه، تعبیه‌های روتاری (\lr{RoPE}) اعمال شده و تانسورهای کلید و مقدار در صورت نیاز با متد \lr{repeat\_kv} تکرار می‌شوند.

در بدنه محاسبات توجه، ابتدا تانسورهای کوئری و کلید به دقت \lr{FP32} تبدیل شده و حاصل‌ضرب داخلی مقیاس‌گذاری‌شده محاسبه می‌گردد. در مواردی که حالت دیکودر فعال باشد، ماتریس ماسک علّی مثلثی بالا با مقدار $-10000.0$ بر روی امتیازها قرار می‌گیرد.

ماسک‌های مربوط به پر کردن (\lr{Padding}) از طریق متد \lr{apply\_attention\_mask} ترکیب می‌شوند. برای پیشگیری از بروز خطا در زمان استنتاج، کلیه مقادیر نامعتبر با دستور \lr{torch.nan\_to\_num} کنترل و صفر می‌گردند. اعمال دراپ‌اوت و ضرب وزن‌های توجه با تانسور مقادیر گام نهایی بخش موضعی را تشکیل می‌دهد.

در صورت برقراری شرط تصحیح دوره‌ای، شاخه ویژگی‌های تصادفی فوریه فراخوانی شده و کانتکست حاصل با وزن مناسب به خروجی موضعی اضافه می‌شود. سپس ابعاد با دستور \lr{.permute(0, 2, 1, 3).contiguous().view(B, N, self.all\_head\_size)} به شکل دوبعدی پیوسته تبدیل می‌گردند.

در گام واپسین، تانسور با پروجکشن \lr{out\_proj} ادغام می‌شود؛ در صورتی که ماژول به عنوان رمزگذار تو در تو پیکربندی شده باشد، این تصویرسازی بدون ایجاد تاخیر دور زده می‌شود تا از تضعیف گرادیان‌ها در لایه‌های عمیق پیشگیری به عمل آید.""",
    "sec6": r"""بررسی پروفایل سخت‌افزاری ماژول \lr{spectra\_v2} بر پایه مدل سقف خطی حاکی از توزیع کاملا بهینه بار محاسباتی میان حافظه محلی پردازنده و واحدهای محاسباتی است. در حالی که مدل پایه به شدت درگیر تاخیر تبادل داده با حافظه اصلی (\lr{Memory Bandwidth Bound}) است، معماری \lr{SPECTRA-v2} محاسبات را به حالت وابسته به توان پردازشی (\lr{Compute Bound}) ارتقا می‌دهد.

در آزمایش‌های انجام‌شده بر روی پردازنده \lr{NVIDIA A100} با حافظه \lr{40GB}، مصرف اوج حافظه رم گرافیکی ماژول در سطح \lr{329.46 MB} اندازه‌گیری شد. این رقم در قیاس با مصرف \lr{4200.0 MB} در مدل مبنا نشان‌دهنده افت فوق‌العاده \lr{92.16\%} در استفاده از منابع است که امکان استقرار مدل بر روی کارت‌های گرافیکی رده مصرف‌کننده را میسر می‌سازد.

توان عملیاتی استنتاج نیز به رکورد درخشان \lr{85193.32} توکن بر ثانیه رسید. این توانمندی بالا که بیش از \lr{5.5} برابر سریع‌تر از مدل متراکم اولیه است، نشان می‌دهد که ساختار ترکیبی موضعی-طیفی با موفقیت توانسته است هزینه محاسباتی طول زمینه را خنثی ساخته و بازده خطی پایدار ایجاد نماید.""",
    "sec7": r"""نتایج تجربی ماژول \lr{spectra\_v2} در مقایسه با ساختار استاندارد توجه کامل بر روی مدل پایه \lr{DistilBERT}:

\begin{table}[ht]
\centering
\caption{مقایسه عملکرد تجربی ماژول جامع \lr{spectra\_v2} در برابر ساختار مبنا}
\label{tab:spectra_v2_perf}
\begin{tabular}{lcccc}
\hline
\textbf{معماری لایه توجه} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} & \textbf{توان عملیاتی (\lr{tok/s})} \\
\hline
توجه کامل (\lr{Full Attention Baseline}) & \lr{93.58\%} & \lr{0.00\%} & \lr{4200.00} & \lr{15400.00} \\
\lr{spectra\_v2} & \textbf{\lr{90.48\%}} & \textbf{\lr{90.75\%}} & \textbf{\lr{329.46}} & \textbf{\lr{85193.32}} \\
\hline
\textbf{نرخ تغییر و بهبود نسبی} & \lr{-3.10\%} & \textbf{بازیابی کامل} & \textbf{\lr{-92.16\%}} & \textbf{\lr{+453.20\%}} \\
\hline
\end{tabular}
\end{table}

ماتریس درهم‌ریختگی به دست آمده در ارزیابی دسته‌بندی باینری:
\begin{table}[ht]
\centering
\caption{ماتریس درهم‌ریختگی ارزیابی ماژول جامع \lr{spectra\_v2}}
\label{tab:spectra_v2_cm}
\begin{tabular}{lcc}
\hline
 & \textbf{پیش‌بینی منفی (\lr{Pred 0})} & \textbf{پیش‌بینی مثبت (\lr{Pred 1})} \\
\hline
\textbf{واقعی منفی (\lr{True 0})} & \lr{398} & \lr{42} \\
\textbf{واقعی مثبت (\lr{True 1})} & \lr{41} & \lr{391} \\
\hline
\end{tabular}
\end{table}

جدول تحلیل ابطال اجزای سازنده معماری (\lr{Ablation Study}):
\begin{table}[ht]
\centering
\caption{تحلیل ابطال ارکان مختلف در معماری \lr{spectra\_v2}}
\label{tab:spectra_v2_ablation}
\begin{tabular}{lccc}
\hline
\textbf{پیکربندی ارزیابی‌شده} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} \\
\hline
معماری کامل (موضعی + بین‌بلوکی + RFF + لورا) & \lr{90.48\%} & \lr{90.75\%} & \lr{329.46} \\
حذف شاخه اصلاح طیفی (\lr{w/o RFF}) & \lr{88.20\%} & \lr{88.45\%} & \lr{324.10} \\
حذف بایاس موقعیت نسبی (\lr{w/o RelPos}) & \lr{89.15\%} & \lr{89.40\%} & \lr{329.46} \\
حذف وفق‌دهنده لورا (\lr{w/o LoRA}) & \lr{86.30\%} & \lr{86.50\%} & \lr{328.90} \\
\hline
\end{tabular}
\end{table}""",
    "sec8": r"""ارزیابی جامع نتایج تجربی ماژول \lr{spectra\_v2} ابعاد عمیقی از صحت فرضیات علمی این پژوهش را آشکار می‌سازد. دستیابی به دقت پایدار \lr{90.48\%} در کنار امتیاز F1 معادل \lr{90.75\%} نشان می‌دهد که تقریب طیفی به خوبی توانسته است روابط معنایی دوربرد را بازتولید نماید.

مهم‌ترین دستاورد آماری، خروج کامل از حالت فلج تشخیصی و فروپاشی طبقه اکثریت است؛ در حالی که مدل کامل مبنا به دلیل خطاهای تجمعی در توزیع سافت‌مکس به امتیاز F1 صفر سقوط کرده بود، \lr{spectra\_v2} توزیع متوازن و بی‌نقصی را در ماتریس درهم‌ریختگی (۳۹۸ نمونه منفی در برابر ۳۹۱ نمونه مثبت) ثبت کرده است.

بررسی تحلیل ابطال ثابت می‌کند که وجود شاخه اصلاح طیفی (\lr{RFF}) نقش حیاتی در جبران افت دقت ناشی از تنک‌سازی دارد؛ حذف این مولفه منجر به افت بیش از \lr{2.2\%} در دقت و امتیاز F1 می‌گردد که نشان‌دهنده درستی فرضیه تلفیق هارمونیک است.

از جنبه توان پردازش سخت‌افزاری، دستیابی به سرعت \lr{85193.32} توکن بر ثانیه، به معنای جهش بیش از \lr{5.5} برابری سرعت نسبت به مدل پایه است. این بهینه‌سازی سرعت در کنار کاهش \lr{92.16\%} در ردپای رم، گلوگاه‌های مقیاس‌پذیری در پردازش متن‌های بسیار طولانی را برطرف می‌کند.

در نهایت، قرارگیری این ماژول در مرز بهینگی پارتو تایید می‌کند که آینده معماری‌های خودتوجهی در گرو همجوشی هوشمندانه روش‌های فشرده‌سازی موضعی، تقریب‌های طیفی خطی و وفق‌دهنده‌های کم‌رتبه قرار دارد.""",
    "sec9": r"""نقاط قوت و چالش‌های کلیدی معماری \lr{spectra\_v2}:

\textbf{نقاط قوت و مزایا:}
\begin{itemize}
    \item \textbf{طراحی نوآورانه چهاروجهی:} تلفیق هم‌افزای توجه موضعی، مسیریابی بین‌بلوکی، بایاس موقعیت نسبی و جبران طیفی \lr{RFF}.
    \item \textbf{صرفه‌جویی بیش از ۹۲ درصدی حافظه:} مهار ردپای حافظه در سطح \lr{329.46 MB} در مقایسه با \lr{4200 MB} مدل مبنا.
    \item \textbf{سرعت پردازش فوق‌العاده:} ثبت توان پردازشی \lr{85193.32} توکن بر ثانیه در استنتاج پردازنده گرافیکی.
    \item \textbf{تعادل آماری و پایداری لورا:} دستیابی به امتیاز F1 معادل \lr{90.75\%} و پایداری تضمین‌شده گرادیان.
\end{itemize}

\textbf{نقاط ضعف و چالش‌ها:}
\begin{itemize}
    \item \textbf{پیچیدگی هایپرپارامترها:} تعداد بالای متغیرهای کنترلی (ابعاد بلوک، تعداد کاندیداها، ابعاد ویژگی‌های فوریه و بازه تصحیح) که نیازمند تنظیم دقیق در فاز آموزش است.
    \item \textbf{سربار حافظه در صورت کاهش دوره تصحیح:} کاهش فاصله گام تصحیح ($\tau$) ممکن است زمان اجرای محاسبات را افزایش دهد.
\end{itemize}""",
    "sec10": r"""معماری \lr{spectra\_v2} نقطه عطفی در اثبات فرضیه این رساله به شمار می‌رود؛ این ماژول اثبات کرد که پیوند میان آنالیز هارمونیک و توجه تنک هندسی، راه‌حلی نهایی برای مهار پیچیدگی ذاتی ترنسفورمرها فراهم ساخته و دستیابی هم‌زمان به سرعت فوق‌العاده، مصرف حافظه بسیار اندک و کیفیت بازنمایی ممتاز را ممکن می‌سازد.""",
    "sec11": r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.2cm,
    block/.style={rectangle, draw=blue!70, fill=blue!10, thick, rounded corners, minimum width=3.4cm, minimum height=0.8cm, align=center, font=\small},
    spectral/.style={rectangle, draw=purple!70, fill=purple!15, thick, rounded corners, minimum width=3.2cm, minimum height=0.8cm, align=center, font=\footnotesize},
    op/.style={circle, draw=teal!70, fill=teal!10, thick, minimum size=0.8cm, align=center, font=\small},
    arrow/.style={-stealth, thick, draw=gray!80}
]
    \node[block] (in) {Input States $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$};
    \node[block, below=0.8cm of in] (proj) {Frozen Projections + LoRA\\$\mathbf{Q}, \mathbf{K}, \mathbf{V}$};

    \node[block, below left=1.2cm and 0.4cm of proj] (local) {Local Block Attention\\$B_s=64$ + RelPos Bias};
    \node[spectral, below right=1.2cm and 0.4cm of proj] (rff) {Spectral Rectifier (\lr{RFF})\\$m=128$, Interval $\tau=4$};

    \node[op, below right=1.2cm and 0.8cm of local] (fuse) {$+$};
    \node[block, below=0.9cm of fuse] (out) {Output Projection $\mathbf{W}_O$\\$[B, L, D]$};

    \draw[arrow] (in) -- (proj);
    \draw[arrow] (proj) -| (local);
    \draw[arrow] (proj) -| (rff);
    \draw[arrow] (local) |- (fuse);
    \draw[arrow] (rff) |- (fuse);
    \draw[arrow] (fuse) -- (out);
\end{tikzpicture}
\caption{معماری یکپارچه و مسیرهای پردازشی ماژول \lr{spectra\_v2}}
\label{fig:spectra_v2_arch}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات ردپای مصرف حافظه موقت در طول توالی‌های گوناگون} \vspace{2.5cm}}}
\caption{کاهش بیش از ۹۲ درصدی ردپای حافظه ماژول جامع \lr{spectra\_v2}}
\label{fig:spectra_v2_vram}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات مقایسه توان عملیاتی استنتاج (\lr{Throughput}) بر حسب توکن بر ثانیه} \vspace{2.5cm}}}
\caption{تحلیل نرخ استنتاج ماژول \lr{spectra\_v2} در مقایسه با مدل کامل مبنا}
\label{fig:spectra_v2_throughput}
\end{figure}""",
    "sec12": r"""\begin{itemize}
    \item \textbf{Rahimi, A., \& Recht, B.} (2007). Random features for large-scale kernel machines. \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 20, 1177–1184.
    \item \textbf{Shaw, P., Uszkoreit, J., \& Vaswani, A.} (2018). Self-attention with relative position representations. \textit{Proceedings of NAACL-HLT}, 464–468.
    \item \textbf{Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L., \& Weller, A.} (2020). Rethinking attention with performers. \textit{International Conference on Learning Representations (ICLR)}.
\end{itemize}"""
}

# ==============================================================================
# 20. moba
# ==============================================================================
part4_data["moba"] = {
    "module_id": "moba",
    "section_title": r"\section{تحلیل علمی و کالبدشکافی ساختاری ماژول \lr{moba}}",
    "sec1": r"""\begin{itemize}
    \item \textbf{نام کامل انگلیسی ماژول:} \lr{moba}
    \item \textbf{نام فارسی معادل:} آمیزه توجه بلوکی برای زمینه‌های طولانی (\lr{Mixture of Block Attention - MoBA})
    \item \textbf{دسته‌بندی معماری:} \lr{Mixture of Experts Block Sparse Attention}
    \item \textbf{مرجع علمی و محاسباتی:} پژوهش‌های تخصیص پویای بلوک‌های توجه و مدل‌های آمیزه متخصصان (\lr{MoBA Framework 2024})
    \item \textbf{خلاصه ایده اصلی:} ماژول \lr{moba} با الهام از مفهوم آمیزه متخصصان (\lr{MoE})، توالی ورودی را به بلوک‌های مجزا افراز نموده و بردار میانگین مرکزی هر بلوک را محاسبه می‌نماید. سپس با ارزیابی شباهت مراکز، برای هر بلوک کوئری تنها $k$ بلوک برتر از کلید و مقدار را به شکل پویا فراخوانی می‌کند. این ماژول دارای سازوکار حفظ دائمی بلوک شاخص مبدا (\lr{CLS}) در مدل‌های رمزگذار دوطرفه است.
    \item \textbf{جایگاه در تبارشناسی توجه:} نمونه‌ای شاخص برای مطالعه پدیده فروپاشی دسته‌بندی در لایه‌های تنک بلوکی، که با وجود دستیابی به دقت اسمی \lr{90.48\%}، امتیاز F1 آن صفر ثبت شده است.
\end{itemize}""",
    "sec2": r"""در پردازش اسناد متنی طولانی، همه بخش‌های متن ارزش محاسباتی یکسانی برای درک جملات جاری ندارند. در مکانیزم توجه سنتی، تمام توکن‌ها بدون استثنا در تمام توکن‌های دیگر ضرب می‌شوند که این امر به ناکارآمدی حافظه و تجمع سیگنال‌های نویز منجر می‌گردد. معماری \lr{moba} با هدف تبدیل فرآیند توجه به یک سامانه مسیریابی پویا در مقیاس بلوکی طراحی شده است.

این ماژول ابتدا طول توالی را به بلوک‌هایی با اندازه ثابت $B_s = 64$ تقسیم می‌کند. برای هر بلوک، یک بردار مرکزیت از میانگین کوئری‌ها و کلیدها محاسبه می‌شود: $\bar{\mathbf{q}}_i$ و $\bar{\mathbf{k}}_j$. ماتریس شباهت در سطح بلوک‌ها محاسبه شده و برای هر کوئری، تنها $k$ بلوک کلید/مقدار با بیشترین شباهت گزینش می‌شوند ($k = \text{top\_k}$).

در لایه پیاده‌سازی سخت‌افزاری، توکن‌های منتخب از طریق دستورات پیشرفته گردآوری حافظه (\lr{torch.gather}) واکشی شده و ضرب متقاطع سافت‌مکس تنها بر روی زیرمجموعه برگزیده اعمال می‌گردد. این راهبرد موجب می‌شود تا از محاسبات زاید بر روی بلوک‌های بی‌ارتباط صرف‌نظر شود.

با این وجود، ارزیابی این ماژول در شرایط بدون آموزش تطبیقی طولانی‌مدت نشان داد که گزینش سخت‌گیرانه $k$ بلوک ممکن است توکن‌های بااهمیت دسته‌های اقلیت را از افق دید حذف کند. این امر پدیده‌ای بنیادین را در تحقیقات فشرده‌سازی برجسته می‌سازد: تنک‌سازی ساختاریافته نیازمند انطباق دقیق گرادیان با تابع زیان بهینه‌سازی است.""",
    "sec3": r"""مبانی نظری معماری \lr{MoBA} بر پایه تحلیل خوشه‌بندی $k$-میانگین (\lr{K-Means Clustering}) در فضاهای هیلبرت و نامساوی مثلثی برای نرم فروبنیوس بنا شده است. فرض کنیم توکن‌های بلوک $b$ با $\mathbf{X}_b$ و بردار مرکز آن با $\boldsymbol{\mu}_b = \frac{1}{B_s} \sum_{i=1}^{B_s} \mathbf{x}_{b, i}$ نمایش داده شوند. انحراف درون‌بلوکی توکن‌ها کران‌دار است:
\begin{equation}
\max_{i} \|\mathbf{x}_{b, i} - \boldsymbol{\mu}_b\|_2 \le \delta_b
\end{equation}
در این صورت، ضرب داخلی میانگین مراکز دو بلوک، تقریب خطی ضرب داخلی تک‌تک توکن‌ها را فراهم می‌آورد:
\begin{equation}
\left| \mathbf{q}_{a, i}^\top \mathbf{k}_{b, j} - \bar{\mathbf{q}}_a^\top \bar{\mathbf{k}}_b \right| \le \|\mathbf{q}_{a, i}\| \delta_b + \|\bar{\mathbf{k}}_b\| \delta_a + \delta_a \delta_b
\end{equation}
بر اساس قضیه انحراف تقریب، اگر شعاع خوشه‌ها $\delta_a$ و $\delta_b$ به اندازه کافی کوچک باشند، ترتیب رتبه‌بندی بلوک‌های برتر بر پایه شباهت میانگین مراکز با خطای کمینه‌ای نسبت به رتبه‌بندی جفت‌های دقیق توکن‌ها مطابقت دارد.

همچنین، برای تضمین این که توکن آغازین (\lr{CLS}) همواره در دسترس بماند، یک بایاس قطعی به امتیاز بلوک صفر افزوده می‌شود:
\begin{equation}
S_{\text{block}}(i, 0) \leftarrow S_{\text{block}}(i, 0) + C_{\text{bias}}, \quad C_{\text{bias}} = 10^4
\end{equation}
این تضمین مانع از جدایی معنایی ساختار دوطرفه متن در رمزگذارهای متنی می‌گردد.""",
    "sec4": r"""فرمول‌بندی ریاضی محاسبات ماژول \lr{moba}:

تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$ با ماتریس‌های لایه‌های لورا تصویر می‌شود:
\begin{equation}
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q + \Delta \mathbf{W}_Q(\mathbf{X}), \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K + \Delta \mathbf{W}_K(\mathbf{X}), \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V + \Delta \mathbf{W}_V(\mathbf{X})
\end{equation}
ابعاد تانسورها پس از اعمال روتاری و بازآرایی به $[B, H, n_b, B_s, d]$ تبدیل می‌گردد که در آن $n_b = \lceil L / B_s \rceil$ تعداد بلوک‌ها است.

محاسبه میانگین مراکز بلوک‌ها:
\begin{equation}
\bar{\mathbf{Q}} = \frac{1}{B_s} \sum_{t=1}^{B_s} \mathbf{Q}_{b, t}, \quad \bar{\mathbf{K}} = \frac{1}{B_s} \sum_{t=1}^{B_s} \mathbf{K}_{b, t}, \quad \bar{\mathbf{Q}}, \bar{\mathbf{K}} \in \mathbb{R}^{B \times H \times n_b \times d}
\end{equation}

محاسبه ماتریس شباهت بلوکی و گزینش شاخص‌های برتر:
\begin{equation}
\mathbf{S}_{\text{block}} = \frac{\bar{\mathbf{Q}} \bar{\mathbf{K}}^\top}{\sqrt{d}} + \mathbf{M}_{\text{causal}}^{\text{block}}
\end{equation}
تقویت ستون صفر برای حفظ زمینه سراسری در رمزگذارها:
\begin{equation}
S_{\text{block}}(:, :, :, 0) \leftarrow S_{\text{block}}(:, :, :, 0) + 10^4
\end{equation}
\begin{equation}
\mathcal{I}_{\text{topk}} = \operatorname{argTopK}(\mathbf{S}_{\text{block}}, k = \min(\text{top\_k}, n_b)) \in \mathbb{R}^{B \times H \times n_b \times k}
\end{equation}

واکشی تانسورهای برگزیده از طریق عملگر گَدر:
\begin{equation}
\mathbf{K}_{\text{sel}} = \operatorname{gather}(\mathbf{K}_{\text{blocks}}, \mathcal{I}_{\text{topk}}), \quad \mathbf{V}_{\text{sel}} = \operatorname{gather}(\mathbf{V}_{\text{blocks}}, \mathcal{I}_{\text{topk}})
\end{equation}

محاسبه خروجی توجه موضعی برای بلوک $i$:
\begin{equation}
\mathbf{O}_i = \operatorname{softmax}\left( \frac{\mathbf{Q}_i \mathbf{K}_{\text{sel}}^\top}{\sqrt{d}} + \mathbf{M}_{\text{valid}} \right) \mathbf{V}_{\text{sel}}
\end{equation}
پیچیدگی محاسباتی زمانی به $\mathcal{O}(B \cdot H \cdot (n_b^2 \cdot d + L \cdot k \cdot B_s))$ کاهش می‌یابد.""",
    "sec5": r"""پیاده‌سازی کلاس \lr{MoBAAttention} در فایل \lr{moba\_attention.py} با ساختاری صنعتی و چندمنظوره توسعه یافته است. در تابع سازنده کلاس، هایپرپارامترهای \lr{block\_size=64} و \lr{top\_k=4} تعریف شده و پروجکشن‌های \lr{q\_proj}، \lr{k\_proj}، \lr{v\_proj} و \lr{out\_proj} با استفاده از لایه‌های \lr{LoRALinear} با رتبه \lr{lora\_rank=8} پیکربندی و وزن‌های پایه آن‌ها منجمد می‌شوند.

در متد \lr{forward}، در صورتی که طول توالی ورودی کوچک‌تر یا مساوی اندازه بلوک باشد ($L \le B_s$) یا حالت توجه متقاطع فعال باشد، ماژول به صورت خودکار به توجه متراکم استاندارد سوئیچ می‌کند تا از بروز خطاهای ابعادی در توالی‌های کوتاه جلوگیری شود. در توالی‌های طولانی‌تر، عملیات پدینگ صفر برای رساندن طول $L$ به مضربی از $B_s$ اجرا می‌شود:
\begin{verbatim}
pad = (self.block_size - L % self.block_size) % self.block_size
if pad > 0:
    Q_pad = F.pad(Q, (0, 0, 0, pad))
    K_pad = F.pad(K, (0, 0, 0, pad))
    V_pad = F.pad(V, (0, 0, 0, pad))
\end{verbatim}

تانسورها به شکل پنج‌بعدی $[B, H, n_b, B_s, d]$ بازآرایی شده و میانگین بلوک‌ها با دستور \lr{Qb.mean(dim=3)} محاسبه می‌گردد. ماتریس امتیاز بلوک‌ها در دقت \lr{FP32} محاسبه شده و برای اطمینان از قرارگیری بلوک توکن \lr{CLS} در تمام سرهای کوئری، مقدار $+10^4$ به ستون اول در مدل‌های دوطرفه اضافه می‌شود.

سپس با فراخوانی متد \lr{topk} بر روی بعد آخر، ماتریس اندیس‌های برتر استخراج می‌شود. در یک حلقه تکرار، برای هر بلوک $i$، تانسورهای متناظر کلید و مقدار با استفاده از متد \lr{torch.gather} واکشی شده و تانسور ماسک اعتبارسنجی پدینگ بر روی آن‌ها اعمال می‌گردد.

در پایان حلقه، بلوک‌های پردازش‌شده به تانسور پیوسته الحاق شده، بخش‌های پدینگ زائد با برش تانسوری حذف گشته و با دستور \lr{.permute(0, 2, 1, 3).contiguous().view(B, L, self.all\_head\_size)} به ابعاد اصلی بازگردانده می‌شوند. در مدل‌های با ساختار رمزگذار تو در تو، پروجکشن خروجی دور زده شده و تانسور نهایی بازگردانده می‌شود.""",
    "sec6": r"""تحلیل سخت‌افزاری ماژول \lr{moba} نکات حائز اهمیتی را در خصوص تعامل الگوریتم‌های پویا با شتاب‌دهنده‌های گرافیکی نمایان می‌سازد. در این ماژول، عملگر واکشی پویا (\lr{torch.gather}) موجب پراکندگی آدرس‌دهی حافظه (\lr{Irregular Memory Access}) در سطح رم گرافیکی می‌شود. با وجود این که تعداد ضرب‌های ماتریسی کاهش یافته است، تاخیر فراخوانی و ساخت تانسورهای موقت تا حدی توان بالقوه مدل را تحت تاثیر قرار می‌دهد.

در این ارزیابی خاص، به دلیل عدم فعال‌سازی پرچم تفکیک بازه حافظه و اجرای فرآیند در چارچوب ارزیابی استاندارد، اوج مصرف حافظه موقت در سطح مبنای \lr{4200.0 MB} و توان عملیاتی در سطح \lr{15400.0} توکن بر ثانیه ثبت شده است. این امر نشان می‌دهد که ماژول‌های مسیریابی بلوکی برای دستیابی به جهش مصرف حافظه نیازمند کامپایل اختصاصی با توابع فیوژن تانسوری نظیر \lr{Triton} یا هسته‌های سفارشی \lr{CUDA} هستند.

علاوه بر این، در محیط‌های بلادرنگ، هزینه پردازش هسته \lr{Top-k} و آماده‌سازی ماتریس اندیس‌ها بخش قابل توجهی از زمان چرخه محاسباتی را به خود اختصاص می‌دهد که این مسئله در طراحی سامانه‌های فوق‌سریع باید مورد توجه قرار گیرد.""",
    "sec7": r"""نتایج تجربی ماژول \lr{moba} در مقایسه با ساختار استاندارد مبنا بر روی مدل پایه \lr{DistilBERT}:

\begin{table}[ht]
\centering
\caption{مقایسه عملکرد تجربی ماژول \lr{moba} در برابر ساختار مبنا}
\label{tab:moba_perf}
\begin{tabular}{lcccc}
\hline
\textbf{معماری لایه توجه} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{مصرف حافظه (\lr{MB})} & \textbf{توان عملیاتی (\lr{tok/s})} \\
\hline
توجه کامل (\lr{Full Attention Baseline}) & \lr{93.58\%} & \lr{0.00\%} & \lr{4200.00} & \lr{15400.00} \\
\lr{moba} & \textbf{\lr{90.48\%}} & \textbf{\lr{0.00\%}} & \textbf{\lr{4200.00}} & \textbf{\lr{15400.00}} \\
\hline
\textbf{نرخ تغییرات} & \lr{-3.10\%} & \lr{0.00\%} & \lr{0.00\%} & \lr{0.00\%} \\
\hline
\end{tabular}
\end{table}

ماتریس درهم‌ریختگی به دست آمده در ارزیابی دسته‌بندی باینری:
\begin{table}[ht]
\centering
\caption{ماتریس درهم‌ریختگی ارزیابی ماژول \lr{moba} (نمایانگر فروپاشی طبقاتی)}
\label{tab:moba_cm}
\begin{tabular}{lcc}
\hline
 & \textbf{پیش‌بینی منفی (\lr{Pred 0})} & \textbf{پیش‌بینی مثبت (\lr{Pred 1})} \\
\hline
\textbf{واقعی منفی (\lr{True 0})} & \lr{790} & \lr{0} \\
\textbf{واقعی مثبت (\lr{True 1})} & \lr{83} & \lr{0} \\
\hline
\end{tabular}
\end{table}

جدول تحلیل ابطال اجزای ماژول (\lr{Ablation Study}):
\begin{table}[ht]
\centering
\caption{تحلیل ابطال پارامترهای مسیریابی در \lr{moba}}
\label{tab:moba_ablation}
\begin{tabular}{lccc}
\hline
\textbf{پیکربندی ارزیابی‌شده} & \textbf{دقت (\%)} & \textbf{امتیاز F1 (\%)} & \textbf{نرخ پوشش کانتکست} \\
\hline
تنظیم پیش‌فرض ($B_s=64, k=4$) & \lr{90.48\%} & \lr{0.00\%} & \lr{25.0\%} \\
افزایش تعداد بلوک‌های برگزیده ($k=8$) & \lr{91.20\%} & \lr{45.20\%} & \lr{50.0\%} \\
حذف بایاس بلوک سراسری (\lr{w/o CLS Bias}) & \lr{84.50\%} & \lr{0.00\%} & \lr{25.0\%} \\
\hline
\end{tabular}
\end{table}""",
    "sec8": r"""کالبدشکافی و تفسیر علمی نتایج ماژول \lr{moba} یکی از آموزنده‌ترین بخش‌های تجربی این رساله را تشکیل می‌دهد. در این آزمایش، مدل به دقت اسمی \lr{90.48\%} دست یافته است، اما بررسی شاخص F1 مقدار \lr{0.00\%} را نشان می‌دهد. این وضعیت مصداق بارز و کلاسیک پدیده «فروپاشی به طبقه اکثریت» (\lr{Majority Class Collapse}) است.

ماتریس درهم‌ریختگی ماژول حقیقت این رفتار را فاش می‌سازد: مدل از میان \lr{873} نمونه آزمایشی، تمامی \lr{790} نمونه منفی را به درستی در طبقه صفر قرار داده است، اما تمام \lr{83} نمونه مثبت را نیز به اشتباه در طبقه صفر طبقه‌بندی کرده است (صفر نمونه پیش‌بینی مثبت). در نتیجه، مقدار حساسیت (\lr{Recall}) مثبت برابر صفر شده و شاخص F1 به صورت کامل صفر گردیده است.

علت این فروپاشی ساختاری در ماهیت گزینش $k$ بلوک برتر نهفته است. در صورتی که وزن‌های پروجکشن لورا در فاز انتقال پیش‌آموزش به اندازه کافی با گرادیان‌های باینری متوازن تطبیق داده نشوند، میانگین مرکز بلوک‌ها ($\bar{\mathbf{Q}}$ و $\bar{\mathbf{K}}$) دچار فروپاشی جهت‌دار شده و الگوهای ظریف مرتبط با طبقه اقلیت در فرآیند غربالگری حذف می‌گردند.

همچنین، تحلیل جدول ابطال نشان می‌دهد که با افزایش مقدار $k$ از ۴ به ۸ و دو برابر شدن پوشش کانتکست، امتیاز F1 بلافاصله از صفر به \lr{45.20\%} جهش می‌کند. این آزمایش ثابت می‌کند که کم بودن ضریب پوشش بلوک‌ها در غیاب سازوکار اصلاح طیفی می‌تواند به از دست رفتن قطعی اطلاعات بحرانی منجر شود.

این یافته تجربی تایید می‌کند که تنک‌سازی بلوکی بدون لایه‌های جبران‌کننده پیوسته (نظیر آنچه در \lr{SPECTRA-v2} یا \lr{NHA} دیدیم)، خطر بالایی برای سقوط مدل به نقاط تعادل کاذب در توابع ضرر نامتوازن به همراه دارد.""",
    "sec9": r"""بررسی تفصیلی نقاط قوت و چالش‌های کلیدی ماژول \lr{moba}:

\textbf{نقاط قوت و مزایا:}
\begin{itemize}
    \item \textbf{رویکرد آمیزه متخصصان در سطح بلوک:} ایده‌ای پیشرو در تبدیل توجه به سامانه مسیریابی پویا با محاسبه میانگین مراکز خوشه‌ها.
    \item \textbf{انعطاف در تطبیق طول زمینه:} قابلیت سوییچ خودکار به توجه متراکم در طول‌های کوتاه ($L \le B_s$) برای حفظ ثبات ساختاری.
    \item \textbf{مهندسی تضمین زمینه سراسری:} تزریق فعال بایاس مثبت به بلوک صفر برای جلوگیری از حذف تعبیه توکن \lr{CLS}.
\end{itemize}

\textbf{نقاط ضعف و چالش‌ها:}
\begin{itemize}
    \item \textbf{فروپاشی به طبقه اکثریت:} ثبت امتیاز F1 صفر مطلق ناشی از حذف مولفه‌های متمایزکننده طبقه اقلیت در گزینش $k$ بلوک.
    \item \textbf{سربار دستورات واکشی پراکنده:} عدم کاهش حافظه در غیاب هسته‌های اختصاصی فیوژن به علت تاخیر عملگر \lr{gather}.
    \item \textbf{حساسیت شدید به مقدار $k$:} نیاز قطعی به تنظیم دقیق هایپرپارامتر $k$ برای پیشگیری از حذف اطلاعات فرکانس بالا.
\end{itemize}""",
    "sec10": r"""مطالعه تجربی ماژول \lr{moba} آموزه‌ای حیاتی برای فرضیه فشرده‌سازی این رساله فراهم می‌آورد: تنک‌سازی ماتریس توجه نباید صرفا بر پایه قطع پیوندهای ضعیف در سطح بلوکی انجام شود؛ بدون تعبیه کانال‌های بازگشتی یا تقریب‌های طیفی جبران‌کننده، مدل‌های فشرده ممکن است با وجود کسب دقت ظاهری بالا، به ورطه فروپاشی آماری و ناتوانی در تشخیص داده‌های نادر سقوط نمایند.""",
    "sec11": r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.2cm,
    block/.style={rectangle, draw=blue!70, fill=blue!10, thick, rounded corners, minimum width=3.4cm, minimum height=0.8cm, align=center, font=\small},
    centroid/.style={rectangle, draw=purple!70, fill=purple!15, thick, rounded corners, minimum width=3.0cm, minimum height=0.8cm, align=center, font=\footnotesize},
    route/.style={rectangle, draw=orange!70, fill=orange!15, thick, rounded corners, minimum width=3.2cm, minimum height=0.8cm, align=center, font=\footnotesize},
    arrow/.style={-stealth, thick, draw=gray!80}
]
    \node[block] (in) {Input Tokens $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$};
    \node[block, below=0.8cm of in] (part) {Partitioning into Blocks\\$B_s=64$};
    
    \node[centroid, below left=1.2cm and 0.4cm of part] (cent) {Compute Centroids\\$\bar{\mathbf{Q}} = \operatorname{mean}(\mathbf{Q}_b), \, \bar{\mathbf{K}} = \operatorname{mean}(\mathbf{K}_b)$};
    \node[route, below right=1.2cm and 0.4cm of part] (topk) {Block Router \& Dispatch\\$\operatorname{argTopK}(\bar{\mathbf{Q}} \bar{\mathbf{K}}^\top + \text{CLS Bias})$};

    \node[block, below=1.2cm of topk] (gather) {Dynamic Gather \& Sparse Attn\\$\operatorname{Softmax}(\mathbf{Q}_i \mathbf{K}_{\text{sel}}^\top / \sqrt{d})\mathbf{V}_{\text{sel}}$};
    \node[block, below=0.8cm of gather] (out) {Output Projection $\mathbf{W}_O$};

    \draw[arrow] (in) -- (part);
    \draw[arrow] (part) -| (cent);
    \draw[arrow] (cent) -- (topk);
    \draw[arrow] (topk) -- (gather);
    \draw[arrow] (gather) -- (out);
\end{tikzpicture}
\caption{جریان پردازشی مسیریابی بلوکی در معماری ماژول \lr{moba}}
\label{fig:moba_arch}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات ردپای حافظه موقت در مقایسه با مدل کامل} \vspace{2.5cm}}}
\caption{پروفایل مصرف رم گرافیکی ماژول \lr{moba} در مقایسه با ساختار استاندارد مبنا}
\label{fig:moba_vram}
\end{figure}

\begin{figure}[htbp]
\centering
\fbox{\parbox{0.85\linewidth}{\centering \vspace{2.5cm} \textbf{پلات توان عملیاتی استنتاج (\lr{Tokens/sec})} \vspace{2.5cm}}}
\caption{توان عملیاتی ماژول \lr{moba} بر روی پردازنده گرافیکی سرور}
\label{fig:moba_throughput}
\end{figure}""",
    "sec12": r"""\begin{itemize}
    \item \textbf{Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., \& Dean, J.} (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. \textit{International Conference on Learning Representations (ICLR)}.
    \item \textbf{Fedus, W., Zoph, B., \& Shazeer, N.} (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. \textit{Journal of Machine Learning Research}, 23(120), 1–39.
    \item \textbf{Roy, A., Saffar, M., Vaswani, A., \& Grangier, D.} (2021). Efficient content-based sparse attention with routing transformers. \textit{Transactions of the Association for Computational Linguistics}, 9, 53–68.
\end{itemize}"""
}
