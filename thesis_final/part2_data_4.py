# -*- coding: utf-8 -*-

modules_chunk4 = [
    # 22. jamba_info
    r"""
\subsection*{ترانسفورمر پیوندی گلوگاه اطلاعاتی جامبا-اینفو (\lr{Jamba-Info})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری ترانسفورمر \lr{Jamba-Info}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{Jamba-Info} توسعه‌ای تئوریک بر ساختارهای هیبرید ترانسفورمر است که اصل گلوگاه اطلاعاتی (\lr{Information Bottleneck Principle}) را در معماری تلفیقی مامبا-توجه یکپارچه می‌سازد. در مدل‌های پیوندی مقیاس‌بزرگ، تنسورهای میانی انباشته‌شده در لایه‌های بازگشتی و هدهای توجه غالباً دچار افزونگی اطلاعاتی شدید (\lr{Information Redundancy}) می‌شوند که علاوه بر هدررفت ظرفیت پارامترها، منجر به بیش‌برازش در وظایف ظریف می‌گردد. \lr{Jamba-Info} با فرموله‌بندی صریح تبادل اطلاعات میان بازنمایی پنهان و متغیر هدف، اطلاعات نامرتبط را فشرده ساخته و تنها اطلاعات با کمترین انتروپی و بیشترین توان پیش‌بینی را در لایه‌ها منتقل می‌کند.

\textbf{فرمول‌بندی و روابط ریاضی:}
در چارچوب بهینه‌سازی گلوگاه اطلاعاتی، تابع هدف لاگرانژ به شرح زیر تدوین می‌شود:
\begin{equation*}
\mathcal{L}_{\text{IB}} = \mathcal{L}_{\text{task}}(Y, \hat{Y}) + \beta_{\text{info}} \cdot I(X; Z)
\end{equation*}
که در آن $I(X; Z)$ اطلاعات متقابل میان توکن‌های ورودی $X$ و بازنمایی فشرده میانی $Z$ است. تنسور ورودی ابتدا از یک نگاشت گلوگاهی تصادفی یا رتبه-پایین عبور می‌کند:
\begin{equation*}
\mu_z = X W_\mu, \quad \log \sigma_z^2 = X W_\sigma
\end{equation*}
\begin{equation*}
Z = \mu_z + \sigma_z \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\end{equation*}
سپس بردار پالایش‌شده $Z$ به طور همزمان به بلوک‌های فضای حالت خطی \lr{SSM} و هدهای توجه چندسره توزیع می‌گردد:
\begin{equation*}
Y_{\text{ssm}} = \operatorname{MambaLayer}(Z), \quad Y_{\text{attn}} = \operatorname{AttentionLayer}(Z)
\end{equation*}
تلفیق خروجی با وزن‌دهی اطلاعاتی و عبور از لایه متخصصان تنک انجام می‌پذیرد:
\begin{equation*}
Y = \operatorname{MoE}\left( \alpha_{\text{info}} \odot Y_{\text{ssm}} + (1 - \alpha_{\text{info}}) \odot Y_{\text{attn}} \right)
\end{equation*}

\textbf{تحلیل کارایی و رفتار بازنمایی:}
اعمال اصل گلوگاه اطلاعاتی در \lr{Jamba-Info} از هدررفت پهنای باند حافظه در ذخیره‌سازی ویژگی‌های زائد جلوگیری کرده و افزونگی پارامتری میان بلوک‌های متخصصان را تا ۳۰ درصد می‌کاهد. این سازوکار به تعمیم‌پذیری بالاتر و استنتاج سریع‌تر بر روی دامنه‌های متنی ناآشنا منجر می‌گردد.
""",

    # 23. kda_attention
    r"""
\subsection*{توجه واپاشی هسته‌بندی‌شده (\lr{Kernelized Decay Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{توجه واپاشی هسته‌بندی‌شده (\lr{KDA})}

\textbf{مبانی نظری و انگیزه معماری:}
معماری توجه واپاشی هسته‌بندی‌شده (\lr{Kernelized Decay Attention - KDA})، پیوندی ژرف میان نظریه تقریب هسته‌ای (\lr{Kernel Approximation}) و دینامیک‌های واپاشی وابسته به زمان (\lr{Temporal Decay Dynamics}) برقرار می‌سازد. در حالی که توجه خطی مبتنی بر هسته پیچیدگی را به سطح خطی می‌رساند، فاقد توانایی طبیعی در اولویت‌بندی بافت‌های نزدیک‌تر در مقایسه با گذشته دور است. معماری \lr{KDA} با ادغام یک فاکتور واپاشی پیوسته و جهت‌دار در ماتریس هسته، امکان تفکیک توکن‌ها بر اساس فاصله مکانی را فراهم کرده و پایداری دینامیکی سیستم بازگشتی را تضمین می‌نماید.

\textbf{فرمول‌بندی و روابط ریاضی:}
با بکارگیری نگاشت هسته‌ای مثبت $\phi(x) \ge 0$ (نظیر فعال‌سازی نامنفی $\operatorname{ELU}+1$)، ماتریس توجه وابسته به فاصله با ضریب واپاشی $\lambda \in (0, 1)$ به صورت زیر تعریف می‌شود:
\begin{equation*}
A_{i, j} = \phi(Q_i)^\top \phi(K_j) \cdot \lambda^{i - j}, \quad \forall i \ge j
\end{equation*}
این فرمول‌بندی را می‌توان به فرم بازگشتی خطی دقیق در طول زمان نگاشت کرد:
\begin{equation*}
\mathbf{S}_t = \lambda \mathbf{S}_{t-1} + \phi(K_t) V_t^\top \in \mathbb{R}^{d_k \times d_v}
\end{equation*}
\begin{equation*}
\mathbf{z}_t = \lambda \mathbf{z}_{t-1} + \phi(K_t) \in \mathbb{R}^{d_k}
\end{equation*}
بازخوانی بردار خروجی برای توکن ورودی فعلی به شکل زیر نرمال‌سازی و استخراج می‌گردد:
\begin{equation*}
Y_t = \frac{\mathbf{S}_t^\top \phi(Q_t)}{\mathbf{z}_t^\top \phi(Q_t) + \epsilon} \in \mathbb{R}^{d_v}
\end{equation*}
در مدل‌های پردازش دوطرفه، فرآیند فوق به صورت متقارن در جهت معکوس نیز اجرا شده و خروجی دو جریان میانگین‌گیری می‌شود.

\textbf{سازوکار حافظه و پیچیدگی:}
پیچیدگی محاسباتی زمانی کل دنباله $\mathcal{O}(L \cdot d_k \cdot d_v)$ است که با طول دنباله به طور اکید خطی رشد می‌کند. در حین استنتاج، هیچ نیازی به نگهداری ماتریس‌های گذشته نبوده و تنها ماتریس حالت $\mathbf{S}_t$ با ابعاد کوچک و ثابت در حافظه مقیم می‌شود.
""",

    # 24. leka_v1
    r"""
\subsection*{توجه هسته‌ای کارآمد زمان-خطی نسخه ۱ (\lr{LEKA v1})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{LEKA v1}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{LEKA v1 (Linear-time Efficient Kernel Attention)} مکانیزم توجه استاندارد را با نگاشت‌های ویژگی تصادفی (\lr{Random Feature Mappings}) و انباشت حالت انجمنی (\lr{Associative State Accumulation}) جایگزین می‌نماید. انگیزه اساسی \lr{LEKA v1}، حذف وابستگی مربعی زمان و حافظه به طول دنباله از طریق تقریب ضرب داخلی هسته در فضای ویژگی‌های بسط‌یافته است. با بهره‌گیری از نگاشت‌های غیرمنفی پایدارشده با نرمال‌سازی لایه‌ای، این ماژول پایداری عددی گرادیان را در طول فرآیند آموزش تضمین می‌کند.

\textbf{فرمول‌بندی و روابط ریاضی:}
به منظور تضمین تقریب نامنفی هسته بدون بروز ناپایداری در مخرج کسر، نگاشت ویژگی به شرح زیر تدوین می‌گردد:
\begin{equation*}
\phi(\mathbf{x}) = \frac{\operatorname{ELU}\left(\operatorname{LayerNorm}(\mathbf{x}) \mathbf{W}_{\text{proj}}\right) + 1.0}{\sqrt{F}} \in \mathbb{R}^{B \times H \times N \times F}
\end{equation*}
که در آن $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{H \times d_k \times F}$ ماتریس بافر پرتاب‌های تصادفی تثبیت‌شده با بعد ویژگی $F = 256$ است. ماتریس حالت سراسری تداعی‌گر از طریق ضرب خارجی ویژگی‌های کلید و مقدار در سراسر دنباله انباشته می‌شود:
\begin{equation*}
\mathbf{S} = \sum_{j=1}^N \phi(K_j) \otimes V_j \in \mathbb{R}^{H \times F \times d_v}
\end{equation*}
\begin{equation*}
\mathbf{z} = \sum_{j=1}^N \phi(K_j) \in \mathbb{R}^{H \times F}
\end{equation*}
سپس بردار زمینه برای هر توکن پرس‌وجو به صورت تحلیلی بازخوانی می‌گردد:
\begin{equation*}
Y_i = \frac{\mathbf{S}^\top \phi(Q_i)}{\mathbf{z}^\top \phi(Q_i) + \epsilon} \in \mathbb{R}^{H \times d_v}
\end{equation*}
\begin{equation*}
\text{Output} = \operatorname{Concat}(Y_1, \dots, Y_H) W_O + b_O \in \mathbb{R}^{B \times N \times D}
\end{equation*}

\textbf{تحلیل کارایی و پیچیدگی:}
پیچیدگی محاسباتی زمانی مدل $\mathcal{O}(N \cdot F \cdot d)$ و پیچیدگی حافظه آن $\mathcal{O}(F \cdot d)$ است که هر دو به طور قطعی مستقل از توان دوم طول دنباله می‌باشند. این ساختار امکان پردازش دسته‌های بسیار بزرگ از توالی‌های طولانی را بدون اشغال حافظه کارت گرافیک میسر می‌سازد.
""",

    # 25. leka_v2
    r"""
\subsection*{توجه هسته‌ای نشانه‌محور با تقطیر دو مقیاسی نسخه ۲ (\lr{LEKA v2})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{LEKA v2}}

\textbf{مبانی نظری و انگیزه معماری:}
اگرچه مدل‌های ساده توجه خطی با خلاصه‌سازی کل دنباله در یک ماتریس واحد سرعت بالایی ارائه می‌دهند، اما در ثبت قله‌های معنایی شدید و جزئیات دستوری محلی دچار ابهام می‌شوند. معماری \lr{LEKA v2 (Landmark-Enhanced Kernel Attention with Dual-Scale Distillation)} این چالش را با ادغام سه مؤلفه نوآورانه حل می‌کند:
\begin{enumerate}
    \item پرتاب‌های ویژگی تصادفی دو مقیاسی (\lr{Dual-Scale Random Features}) جهت پوشش همزمان فرکانس‌های طیفی گوناگون.
    \item تقطیر نشانه‌های شاخص آموزش‌پذیر (\lr{Learned Landmark Tokens}) جهت استخراج نقاط عطف معنایی با لنگراندازی اجباری توکن شاخص کلاس (\lr{[CLS] Anchor}).
    \item شاخه توجه موضعی پنجره لغزان مجهز به گیتینگ تطبیقی جهت حفظ الگوهای نحوی همسایه.
\end{enumerate}

\textbf{فرمول‌بندی و روابط ریاضی:}
ورودی‌ها ابتدا به دو پایگاه ویژگی متعامد با ابعاد $F_1 = 128$ و $F_2 = 64$ تصویر می‌شوند:
\begin{equation*}
\phi_1(\mathbf{x}) = \frac{\left[\cos(\mathbf{x} \mathbf{W}_1), \; \sin(\mathbf{x} \mathbf{W}_1)\right]}{\sqrt{F_1}}, \quad \phi_2(\mathbf{x}) = \frac{\left[\cos(\mathbf{x} \mathbf{W}_2), \; \sin(\mathbf{x} \mathbf{W}_2)\right]}{\sqrt{F_2}}
\end{equation*}
\begin{equation*}
\phi(\mathbf{x}) = \left[\phi_1(\mathbf{x}), \; \phi_2(\mathbf{x})\right] \in \mathbb{R}^{B \times H \times N \times (2F_1 + 2F_2)}
\end{equation*}
سپس یک شبکه امتیازدهی پرسپترون چندلایه، امتیاز برجستگی هر توکن را ارزیابی می‌کند، در حالی که توکن شاخص کلاس با امتیاز بسیار بزرگ $s_0 = 10^6$ به طور قطعی به عنوان نشانه اصلی تثبیت می‌گردد:
\begin{equation*}
s = \operatorname{MLP}_{\text{scorer}}(X), \quad s_0 = 10^6
\end{equation*}
تعداد $L_{\max}$ توکن برتر به عنوان نشانه‌های جهانی انتخاب شده و ماتریس توجه به نشانه‌ها ($C_{\text{leka}}$) بر اساس آن‌ها شکل می‌گیرد. همزمان، شاخه پنجره لغزان محلی خروجی $C_{\text{local}}$ را بر روی نوار $|i - j| \le W$ تولید می‌کند. ادغام نهایی از طریق یک گیت تطبیقی صورت می‌پذیرد:
\begin{equation*}
C = \sigma(g) \odot C_{\text{leka}} + (1 - \sigma(g)) \odot C_{\text{local}}
\end{equation*}
\begin{equation*}
\text{Output} = C W_{\text{out}} + b_{\text{out}}
\end{equation*}

\textbf{تحلیل پیچیدگی:}
این معماری با ترکیب هوشمندانه نشانه‌های متنی جهانی و پنجره محلی، توان بیانی نزدیک به توجه کامل را با هزینه خطی $\mathcal{O}(N \cdot (F + W + L_{\max}))$ در اختیار شبکه قرار می‌دهد.
""",

    # 26. leka_v3
    r"""
\subsection*{توجه هسته‌ای نشانه‌محور آگاه از ارزش با پایایی در برابر پدینگ نسخه ۳ (\lr{LEKA v3})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{LEKA v3}}

\textbf{مبانی نظری و انگیزه معماری:}
در توالی‌های متنی طبیعی که دارای طول‌های متغیر بوده و از توکن‌های پدینگ (\lr{Padding Tokens}) برای یکسان‌سازی دسته‌ها استفاده می‌کنند، مدل‌های نشانه‌محور پیشین با معضل جدی انتخاب ناخواسته توکن‌های بی‌معنی پدینگ در مجموعه نشانه‌ها مواجه بودند که به سرریزهای عددی و خطاهای تولید مقادیر نامعتبر (\lr{NaN}) منجر می‌شد. معماری \lr{LEKA v3} این نقیصه را با معرفی سه نوآوری بنیادین برطرف می‌سازد:
\begin{enumerate}
    \item \textbf{امتیازدهی آگاه از ارزش (\lr{Value-Aware Saliency Scoring}):} به جای امتیازدهی بر اساس کلیدها که صرفاً محل توجه را نشان می‌دهند، بردارهای مقدار ($V$) که حامل محتوای واقعی اطلاعات هستند ارزیابی می‌شوند.
    \item \textbf{فیلترگذاری دقیق ماسک پدینگ:} مسدودسازی قاطع موقعیت‌های پدینگ با امتیاز $-10^4$ و تضمین لنگراندازی توکن \lr{[CLS]} با امتیاز $+10^6$.
    \item \textbf{فعال‌سازی نرم انتخاب (\lr{Softpick Activation}):} گزینش هموار و مشتق‌پذیر نشانه‌ها جهت ممانعت از گسستگی شیب گرادیان.
\end{enumerate}

\textbf{فرمول‌بندی و روابط ریاضی:}
امتیاز برجستگی توکن‌ها مستقیماً از بازنمایی مقادیر استخراج می‌شود:
\begin{equation*}
\mathbf{s} = \operatorname{MLP}_{\text{scorer}}(\mathbf{V}) = \mathbf{W}_2 \operatorname{GELU}(\mathbf{V} \mathbf{W}_1 + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^{B \times H \times N}
\end{equation*}
ماسک پدینگ باینری بر روی امتیازها اعمال گردیده و نشانه کلاس تثبیت می‌شود:
\begin{equation*}
\mathbf{s} = \mathbf{s} \odot M_{\text{pad}} - 10^4 \cdot (1 - M_{\text{pad}}), \quad \mathbf{s}_{:, :, 0} = +10^6
\end{equation*}
سپس برترین نشانه‌ها انتخاب شده و وزن‌های توجه از طریق عملگر نرمال‌سازی با دمای کالیبره استخراج می‌گردند:
\begin{equation*}
P_{\text{land}} = \operatorname{Softpick}(\mathbf{s}, L_{\max}), \quad C_{\text{land}} = P_{\text{land}} V_{\text{land}}
\end{equation*}
ترکیب با بافت محلی پنجره لغزان مجهز به ماسک پدینگ به صورت زیر اجرا می‌شود:
\begin{equation*}
C = \sigma(g) \odot C_{\text{land}} + (1 - \sigma(g)) \odot C_{\text{local}}
\end{equation*}
\begin{equation*}
\text{Output} = C W_{\text{out}} + b_{\text{out}}
\end{equation*}

\textbf{پایداری عددی و راندمان محاسباتی:}
طراحی دقیق ماسک و امتیازدهی مبتنی بر محتوا، خطاهای محاسباتی را به طور کامل ریشه‌کن ساخته و مدلی فوق‌العاده پایدار در محیط‌های آموزشی مختلط (\lr{Mixed-Precision FP16}) ارائه می‌دهد که با پیچیدگی کاملاً خطی $\mathcal{O}(N)$ اجرا می‌گردد.
""",

    # 27. linformer_attention
    r"""
\subsection*{توجه لینفورمر: تصویرسازی دنباله با مرتبه خطی (\lr{Linformer Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه لینفورمر (\lr{Linformer})}

\textbf{مبانی نظری و انگیزه معماری:}
معماری لینفورمر (\lr{Linformer Attention}) که توسط وانگ و همکاران (۲۰۲۰) معرفی شد، یکی از پیشگامانه‌ترین رویکردهای نظری در شکستن سد محاسباتی $\mathcal{O}(L^2)$ ترانسفورمر است. پایه علمی لینفورمر بر این کشف تجربی و اثبات ریاضی استوار است که ماتریس نرمال‌شده توجه $P = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)$ دارای رتبه موثر بسیار پایینی ($k \ll L$) بوده و مقادیر منفرد آن ($\sigma_i(P)$) با نرخ نمایی زوال می‌یابند:
\begin{equation*}
\sigma_i(P) \le c \cdot \exp(-\alpha \cdot i), \quad \alpha > 0
\end{equation*}
بر پایه لم پرآوازه جانسون-لیندن‌استراوس (\lr{Johnson-Lindenstrauss Lemma})، می‌توان با تصویر کردن طول توالی کلید و مقدار بر روی یک بعد بسیار کوچکتر و ثابت $k = \mathcal{O}(d / \epsilon^2)$ با ماتریس‌های پرتاب خطی تصادفی یا آموزش‌پذیر $E, F \in \mathbb{R}^{k \times L}$، بدون از دست دادن دقت بازنمایی، پیچیدگی محاسباتی و حافظه را به $\mathcal{O}(L \cdot k)$ کاهش داد.

\textbf{فرمول‌بندی و روابط ریاضی:}
در لینفورمر، پرس‌وجوها بدون تغییر بعد طول توالی باقی می‌مانند ($Q \in \mathbb{R}^{B \times H \times L \times d}$)، اما توالی کلیدها و مقادیر در امتداد بعد طول دنباله متراکم می‌گردند:
\begin{equation*}
K_{\text{proj}} = E K \in \mathbb{R}^{B \times H \times k \times d}, \quad V_{\text{proj}} = F V \in \mathbb{R}^{B \times H \times k \times d}
\end{equation*}
ماتریس امتیاز توجه با ابعاد مستطیلی فشرده $L \times k$ به شرح زیر ارزیابی می‌شود:
\begin{equation*}
S = \frac{Q K_{\text{proj}}^\top}{\sqrt{d}} \in \mathbb{R}^{B \times H \times L \times k}
\end{equation*}
\begin{equation*}
P = \operatorname{Softmax}(S, \text{dim}=-1) \in \mathbb{R}^{B \times H \times L \times k}
\end{equation*}
سپس بردار زمینه از ضرب در مقادیر فشرده‌شده حاصل می‌گردد:
\begin{equation*}
O = P V_{\text{proj}} \in \mathbb{R}^{B \times H \times L \times d}
\end{equation*}
\begin{equation*}
Y = \operatorname{Concat}(O_1, \dots, O_H) W_O + b_O \in \mathbb{R}^{B \times L \times D}
\end{equation*}
از دیدگاه نظری، توزیع متقارن انرژی تکین ($\sqrt{\Sigma}$) در پرتاب‌های فاکتورسازی‌شده تضمین می‌نماید که گرادیان‌ها در محاسبات با دقت \lr{FP16} دچار انفجار یا محوشدگی نگردند:
\begin{equation*}
E \sim \mathcal{N}\left(0, \; \frac{1}{\sqrt{k}}\right), \quad F \sim \mathcal{N}\left(0, \; \frac{1}{\sqrt{k}}\right)
\end{equation*}

\textbf{تحلیل پیچیدگی و برتری حافظه:}
پیچیدگی زمانی و حافظه لینفورمر از $\mathcal{O}(L^2)$ به $\mathcal{O}(L \cdot k \cdot d)$ تبدیل می‌شود که برای اسناد طولانی که در آن‌ها $L \gg k$ است، مقیاس‌بندی اکیداً خطی با طول متن به ارمغان می‌آورد و نیاز به ماتریس‌های حافظه موقت بزرگ را کاملاً منتفی می‌سازد.
""",

    # 28. lola_attention
    r"""
\subsection*{توجه محلی‌سازی‌شده رتبه-پایین با سیستم حافظه سه‌گانه (\lr{LoLA Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{LoLA Attention}}

\textbf{مبانی نظری و انگیزه معماری:}
در استنتاج خودهمبسته توالی‌های طولانی متنی در مدل‌های زبانی بزرگ، انباشت بی‌پایان بردارهای گذشته در کش \lr{KV} منجر به اشغال بخش اعظم حافظه \lr{HBM} پردازنده و در نتیجه محدود شدن توان پردازش همزمان به دلیل تنگنای پهنای باند حافظه می‌گردد. معماری توجه محلی‌سازی‌شده رتبه-پایین (\lr{Low-Rank Localized Attention - LoLA}) این چالش حیاتی را با الهام از ساختار سلسله‌مراتبی حافظه در سامانه‌های پردازشی حل می‌کند. \lr{LoLA} تاریخچه متن را به سه سیستم کش حافظه چندسطحی افراز می‌نماید:
\begin{enumerate}
    \item \textbf{کش محلی معاصر ($\mathcal{C}_{\text{recent}}$):} یک پنجره لغزان با دقت کامل برای نگهداری توکن‌های با فرکانس بالا در همسایگی مستقیم.
    \item \textbf{کش تنک نمونه‌برداری‌شده ($\mathcal{C}_{\text{sparse}}$):} توکن‌های گذشته با گام‌های منظم جهت حفظ نقاط عطف میانی.
    \item \textbf{کش فشرده رتبه-پایین \lr{SVD} ($\mathcal{C}_{\text{lr}}$):} استخراج مولفه‌های بنیادین از طریق تجزیه مقادیر منفرد مقطوع بر روی تاریخچه بسیار دور.
\end{enumerate}

\textbf{فرمول‌بندی و روابط ریاضی:}
برای تاریخچه توکن‌ها با طول $L$، افراز سه‌گانه حافظه بر روی کلیدها و مقادیر به صورت تحلیلی پیاده‌سازی می‌گردد:
\begin{enumerate}
    \item \textbf{کش محلی:}
    \begin{equation*}
    K_{\text{recent}} = K[:, :, L-W:, :], \quad V_{\text{recent}} = V[:, :, L-W:, :] \in \mathbb{R}^{B \times H \times W \times d}
    \end{equation*}
    \item \textbf{کش تنک با گام نمونه‌برداری $s$:}
    \begin{equation*}
    K_{\text{sparse}} = K[:, :, 0:L-W:s, :], \quad V_{\text{sparse}} = V[:, :, 0:L-W:s, :] \in \mathbb{R}^{B \times H \times L_{\text{sparse}} \times d}
    \end{equation*}
    \item \textbf{کش رتبه-پایین \lr{SVD}:}
    با اعمال تجزیه مقطوع مقادیر منفرد بر روی کلیدهای تاریخچه دور $K_{\text{hist}} = K[:, :, :L-W, :]$:
    \begin{equation*}
    K_{\text{hist}} \approx U_k \Sigma_k V_k^\top \implies K_{\text{lr}} = U_{k, :, :, :k_{\text{rank}}} \Sigma_{k, :, :k_{\text{rank}}} \in \mathbb{R}^{B \times H \times k_{\text{rank}} \times d}
    \end{equation*}
    و به طریق مشابه برای مقادیر $V_{\text{lr}}$.
\end{enumerate}
سپس کلیدها و مقادیر در امتداد بعد طول متصل می‌شوند:
\begin{equation*}
\widetilde{K} = \left[ K_{\text{recent}} \;;\; K_{\text{sparse}} \;;\; K_{\text{lr}} \right], \quad \widetilde{V} = \left[ V_{\text{recent}} \;;\; V_{\text{sparse}} \;;\; V_{\text{lr}} \right]
\end{equation*}
\begin{equation*}
S = \frac{Q \widetilde{K}^\top}{\sqrt{d}} + M_{\text{mask}} \in \mathbb{R}^{B \times H \times L \times L_{\text{total}}}
\end{equation*}
\begin{equation*}
O = \operatorname{Softmax}(S, \text{dim}=-1) \widetilde{V} \in \mathbb{R}^{B \times H \times L \times d}
\end{equation*}
که در آن طول موثر بافت $L_{\text{total}} = W + \lfloor(L-W)/s\rfloor + k_{\text{rank}} \ll L$ است.

\textbf{تحلیل کارایی و صرفه‌جویی حافظه:}
این نوآوری اندازه کش \lr{KV} را از میزان رشد خطی نامحدود $\mathcal{O}(L \cdot d)$ به سقف فشرده $\mathcal{O}((W + L/s + k_{\text{rank}}) \cdot d)$ کاهش می‌دهد. در نتیجه، تنگنای واکشی داده از حافظه \lr{HBM} در زمان تولید متن برطرف شده و سرعت پاسخ‌دهی مدل چندین برابر می‌گردد.
"""
]
print("Chunk 4 loaded successfully, total modules:", len(modules_chunk4))
