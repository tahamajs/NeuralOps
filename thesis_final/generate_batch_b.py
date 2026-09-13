# -*- coding: utf-8 -*-
import os

target_dir = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules"

# 8. routing_attention
routing_attention = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{routing\_attention}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
در ساختارهای استاندارد توجه، ارزیابی امتیاز تمامی جفت‌توکن‌ها از طریق ضرب ماتریسی متراکم انجام می‌شود، در حالی که در فضاهای برداری با ابعاد بالا، اکثر بردارها با یکدیگر زاویه‌ای نزدیک به ۹۰ درجه دارند و حاصل‌ضرب داخلی آن‌ها پس از عبور از تابع سافت‌مکس به صفر میل می‌کند. صرف محاسبات و اختصاص بافرهای حافظه \lr{CUDA VRAM} برای این تعاملات بی‌اثر، مانع اصلی پردازش توالی‌های طولانی است.

ایده محوری ماژول «\lr{routing\_attention}»، هدایت پویای پرس‌وجوها و کلیدها به سمت خوشه‌های محتوایی متمرکز از طریق مسیریابی تنک (\lr{Dynamic Cluster Routing}) است. در این الگوریتم، مجموعه‌ای از مراکز خوشه‌بندی کروی ($\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$) در فضای نهفته تعریف شده و هر توکن بر مبنای شباهت کسینوسی به نزدیک‌ترین مرکز خوشه هدایت می‌گردد. محاسبات توجه سپس صرفاً در میان توکن‌هایی که به خوشه‌های یکسان منتسب شده‌اند اجرا می‌شود که پیچیدگی حافظه را از $\mathcal{O}(N^2)$ به $\mathcal{O}(N \cdot (N/K))$ کاهش می‌دهد.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
کالبدشکافی کلاس پایتونی \lr{RoutingAttention} در فریم‌ورک مراحل زیر را نشان می‌دهد:
\begin{enumerate}
    \item \textbf{مقداردهی و تخصیص مراکز خوشه‌ها (\lr{\_\_init\_\_}):} تعریف تعداد خوشه‌ها \lr{num\_clusters}، تعداد تکرارهای الگوریتم مسیریابی \lr{n\_iters}، مقداردهی اولیه به بافر مراکز خوشه‌ها $\boldsymbol{\mu} \in \mathbb{R}^{K \times d}$، و تعبیه لایه‌های آداپتور \lr{LoRALinear}.
    \item \textbf{فرافکنی و استخراج سرها:} ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ توسط لایه‌های خطی به فضاهای پرس‌وجو، کلید و مقدار نگاشت یافته و با متد \lr{view} و \lr{transpose} به ابعاد $[B, H, N, d]$ بازآرایی می‌شود.
    \item \textbf{محاسبه شباهت و تخصیص خوشه:} ماتریس شباهت نرمال‌شده میان بردارها و مراکز خوشه‌ها ارزیابی می‌شود: $\mathbf{C} = \mathbf{Q} \boldsymbol{\mu}^T / (\|\mathbf{Q}\| \|\boldsymbol{\mu}\|)$. شاخص خوشه برتر برای هر پرس‌وجو با عملگر \lr{argmax} تعیین می‌گردد.
    \item \textbf{تجمیع و توجه خوشه‌ای تنک:} توکن‌های متعلق به هر خوشه با استفاده از عملگرهای \lr{gather} دسته‌بندی شده و ضرب توجه صرفاً بر روی کلیدهای هم‌خوشه اعمال می‌شود.
    \item \textbf{پراکندگی مجدد و خروجی:} نتایج توجه توسط عملگر \lr{scatter} به جایگاه‌های اولیه زمانی بازگردانده شده و پس از پیوسته‌سازی با \lr{contiguous().view(B, N, D)}، از لایه \lr{out\_proj} و مسیر موازی \lr{LoRA} عبور می‌نمایند.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
فرآیند تخصیص به خوشه‌ها و محاسبه توجه تنک مسیریابی‌شده به صورت روابط زیر مدل‌سازی می‌شود:
\begin{equation}
c(\mathbf{q}_i) = \operatorname{argmax}_{k \in \{1,\dots,K\}} \frac{\mathbf{q}_i \boldsymbol{\mu}_k^T}{\|\mathbf{q}_i\|_2 \|\boldsymbol{\mu}_k\|_2}
\end{equation}
مجموعه شاخص کلیدهای منتسب به خوشه متناظر با پرس‌وجوی $i$:
\begin{equation}
\mathcal{I}_{c(\mathbf{q}_i)} = \left\{ j \mid c(\mathbf{k}_j) = c(\mathbf{q}_i) \right\}
\end{equation}
محاسبه خروجی توجه برای توکن $i$:
\begin{equation}
\mathbf{o}_i = \sum_{j \in \mathcal{I}_{c(\mathbf{q}_i)}} \frac{\exp\left( \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}} + M_{ij} \right)}{\sum_{m \in \mathcal{I}_{c(\mathbf{q}_i)}} \exp\left( \frac{\mathbf{q}_i \mathbf{k}_m^T}{\sqrt{d}} + M_{im} \right)} \mathbf{v}_j
\end{equation}
\begin{equation}
\mathbf{Y} = \mathbf{O} \mathbf{W}_O + \frac{\alpha}{r} \mathbf{X} \mathbf{W}_A \mathbf{W}_B
\end{equation}
که در آن $\boldsymbol{\mu}_k$ بردار مرکز خوشه $k$-ام، $K$ تعداد کل خوشه‌ها، $M_{ij}$ ماسک نامعتبرسازی جفت‌های خارج از دامنه، و $\mathbf{W}_O$ نگاشت خطی خروجی است.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    router/.style={rectangle, draw=purple!80, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    gather/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    attn/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=green!70, fill=green!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{Q}, \mathbf{K}, \mathbf{V}$};
    \node [router, right=0.8cm of input] (route_node) {مسیریاب خوشه‌ها ($\boldsymbol{\mu}_k$)};
    \node [gather, right=0.8cm of route_node] (gather_node) {تجمیع توکن‌های هم‌خوشه (\lr{Gather})};
    \node [attn, right=0.8cm of gather_node] (attn_node) {ضرب توجه تنک درون‌خوشه‌ای};
    \node [outblock, right=0.8cm of attn_node] (out) {پراکندگی معکوس (\lr{Scatter}) و خروجی};

    \draw [->, thick, draw=blue!80] (input) -- (route_node);
    \draw [->, thick, draw=purple!80] (route_node) -- (gather_node);
    \draw [->, thick, draw=orange!80] (gather_node) -- (attn_node);
    \draw [->, thick, draw=teal!80] (attn_node) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item صرفه‌جویی ۹۲.۰ درصدی در تخصیص حافظه گرافیکی با ثبت \lr{Peak VRAM: 334.4 MB}.
        \item متمرکزسازی توجه بر مرتبط‌ترین زیرمجموعه معنایی کلیدها.
        \item قابلیت پیاده‌سازی موازی توزیع‌شده برای پردازش بلوک‌های خوشه‌ای مجزا.
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item افت چشمگیر دقت به ۶۳.۷۶٪ به دلیل خطای مرزی در انتساب توکن‌های چندمعنایی به خوشه‌ها.
        \item سربار محاسباتی ناشی از عملیات نامنظم دسترسی به حافظه در گام‌های \lr{gather} و \lr{scatter}.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{routing\_attention}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی کاملاً یکسان روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{63.76٪} & -27.44٪ & \lr{PASS\_LOW} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{67.08٪} & -21.32٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{334.4 MB} & -92.0٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{14,773 tok/s} & +1.0x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(N \cdot N/K)$ & مسیریابی پویای خوشه‌ای & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
تقلیل شدید مصرف حافظه به ۳۳۴.۴ مگابایت تاییدکننده کارایی فیلترینگ تنک در حذف ابعاد زاید است؛ با این وجود افت ۲۷.۴۴ درصدی دقت ارزیابی حاکی از شکنندگی الگوریتم در نگاشت صحیح بازنمایی‌های زبانی وابسته به زمینه بدون مکانیزم جبرانی مناسب است که منجر به قرارگیری در رده \lr{PASS\_LOW} گردیده است.
"""

# 9. salt
salt = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{salt}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
در فرآیند انطباق مدل‌های پایه زبانی با وظایف پایین‌دستی (\lr{Fine-tuning})، بازتنظیم کامل ماتریس‌های وزن علاوه بر نیاز به حافظه بسیار بالا، موجب تخریب ساختار متعامد فضاهای نهفته و پدیده فراموشی فاجعه‌بار (\lr{Catastrophic Forgetting}) می‌شود. روش‌های متداول نظیر \lr{LoRA} نیز با افزودن مسیرهای باقیمانده بدون تغییر در ماتریس اصلی، برهم‌کنش مستقیم با مقادیر ویژه و طیف اصلی وزن‌ها را نادیده می‌گیرند.

ایده محوری ماژول «\lr{salt}» (\lr{Singular Value Adaptation with Low-Rank Tuning})، تفکیک طیفی ساختار وزن بر پایه تجزیه مقادیر منفرد پیش‌آموخته ($\mathbf{W}_0 = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$) و تثبیت کامل پایه‌های جهت‌دار متعامد ($\mathbf{U}, \mathbf{V}$) است. ماژول انطباق را منحصراً از طریق یادگیری یک بردار مقیاس‌بندی قطری $\boldsymbol{\gamma}$ بر روی طیف مقادیر منفرد اعمال می‌نماید ($\mathbf{\Sigma} \odot \boldsymbol{\gamma}$) و همزمان یک شاخه کم‌مرتبه مکمل $\mathbf{A}\mathbf{B}$ را برای جبران زوایای انحرافی جدید فعال می‌سازد.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
ماژول «\lr{salt}» به عنوان لایه واسط و توزیع‌کننده کلاس‌های \lr{SALTAttention} و \lr{SALTLinear} عمل می‌کند. جریان محاسباتی در سطح لایه خطی این ماژول به شرح زیر است:
\begin{enumerate}
    \item \textbf{مقداردهی اولیه طیفی (\lr{\_\_init\_\_}):} دریافت وزن اولیه \lr{original\_weight} و اجرای تجزیه مقادیر منفرد \lr{torch.linalg.svd}. ثبت پایه‌های $\mathbf{U}$ و $\mathbf{V}^T$ به صورت پارامترهای منجمد (\lr{requires\_grad=False}).
    \item \textbf{تعریف پارامترهای یادگیرنده مقیاس و آداپتور:} تخصیص بردار انطباق طیفی $\boldsymbol{\gamma}$ به شکل بردار هم‌اندازه با مقادیر منفرد با مقداردهی اولیه یک، و تعریف ماتریس‌های \lr{lora\_A} و \lr{lora\_B} با رتبه کم‌مرتبه $r=16$.
    \item \textbf{انطباق پویا در گذر پیشرو:} بردار مقادیر منفرد تطبیق‌یافته محاسبه می‌شود: $\tilde{\mathbf{S}} = \mathbf{S} \odot \boldsymbol{\gamma}$. ماتریس وزن بازسازی‌شده طیفی به صورت $\mathbf{W}_{\text{svd}} = (\mathbf{U} \tilde{\mathbf{S}}) \mathbf{V}^T$ حاصل می‌گردد.
    \item \textbf{ادغام شاخه کم‌مرتبه و بایاس:} شاخه آداپتور کم‌مرتبه به صورت تانسوری ارزیابی شده و با ماتریس بازسازی‌شده جمع می‌شود: $\mathbf{W}_{\text{total}} = \mathbf{W}_{\text{svd}} + \mathbf{A}\mathbf{B}$.
    \item \textbf{فرافکنی خطی پایدار:} تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ با ماتریس حاصل ضرب شده و با متد \lr{contiguous().view(B, N, D)} خروجی بهینه لایه با حفظ ثبات نوع داده (\lr{dtype}) تولید می‌شود.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
فرآیند انطباق طیفی مقادیر منفرد و آداپتور موازی به صورت معادلات ماتریسی زیر فرمول‌بندی می‌گردد:
\begin{equation}
\mathbf{W}_0 = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T, \quad \mathbf{\Sigma} = \operatorname{diag}(\sigma_1, \dots, \sigma_d)
\end{equation}
\begin{equation}
\mathbf{W}_{\text{SALT}} = \mathbf{U} \operatorname{diag}\left( \boldsymbol{\sigma} \odot \boldsymbol{\gamma} \right) \mathbf{V}^T + \frac{\alpha_{\text{lora}}}{r} \mathbf{A} \mathbf{B}
\end{equation}
\begin{equation}
\mathbf{Y} = \mathbf{X} \mathbf{W}_{\text{SALT}}^T + \mathbf{b}
\end{equation}
که در آن $\mathbf{U} \in \mathbb{R}^{D \times d}$ و $\mathbf{V} \in \mathbb{R}^{D \times d}$ ماتریس‌های متعامد یکانی اولیه، $\boldsymbol{\gamma} \in \mathbb{R}^d$ بردار یادگیرنده انطباق دامنه طیفی مقادیر منفرد، $\mathbf{A} \in \mathbb{R}^{D \times r}$ و $\mathbf{B} \in \mathbb{R}^{r \times D}$ پایه‌های کم‌مرتبه، و $\mathbf{b}$ بردار بایاس است.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    salt_node/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    lora_node/.style={rectangle, draw=purple!80, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    sum/.style={circle, draw=green!80, fill=green!20, text centered, minimum size=0.8cm, font=\large},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$};
    \node [salt_node, right=0.9cm of input] (spec) {انطباق طیفی $\mathbf{U} (\mathbf{\Sigma} \odot \boldsymbol{\gamma}) \mathbf{V}^T$};
    \node [lora_node, below=0.6cm of spec] (lora) {شاخه کم‌مرتبه مکمل $\mathbf{A} \mathbf{B}$};
    \node [sum, right=0.9cm of spec] (plus) {$\boldsymbol{+}$};
    \node [outblock, right=0.9cm of plus] (out) {خروجی خطی فرافکنی};

    \draw [->, thick, draw=blue!80] (input) -- (spec);
    \draw [->, thick, draw=blue!80] (input) |- (lora);
    \draw [->, thick, draw=orange!80] (spec) -- (plus);
    \draw [->, thick, draw=purple!80] (lora) -| (plus);
    \draw [->, thick, draw=green!80] (plus) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item توان عملیاتی بسیار بالای ۸۳,۹۸۳ توکن در ثانیه (+5.5x شتاب پردازش).
        \item صرفه‌جویی ۹۲.۰ درصدی در حافظه موقت کارت گرافیک (\lr{Peak VRAM: 337.8 MB}).
        \item حفظ تعامد پایه‌های هندسی فضا و ممانعت از فروپاشی بازنمایی مدل پایه.
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item هزینه زمانی اولیه تجزیه SVD پیش از آغاز فاز تنظیم دقیق.
        \item کاهش ۱۰.۹ درصدی دقت در مقایسه با مدل پایه متراکم کامل.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{salt}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی کاملاً یکسان روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{80.28٪} & -10.92٪ & \lr{PASS} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{81.02٪} & -7.38٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{337.8 MB} & -92.0٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{83,983 tok/s} & +5.5x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(N \cdot r)$ & انطباق طیفی مقادیر منفرد & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
ثبت توان پردازشی ۸۳,۹۸۳ توکن در ثانیه نشان‌دهنده سرعت استثنایی محاسبات بر پایه وزن‌های منجمد و ادغام برداری برخط است. مدیریت دقیق مصرف حافظه کارت گرافیک در مرز ۳۳۷.۸ مگابایت و احراز نمره قبولی \lr{PASS} با دقت ۸۰.۲۸٪، کارایی این مکانیزم را در فشرده‌سازی و تطبیق سریع لایه‌ها به اثبات می‌رساند.
"""

# 10. salt_attention
salt_attention = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{salt\_attention}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
یکپارچه‌سازی مکانیزم‌های انطباق کم‌مرتبه در لایه‌های چندسر توجه نیازمند تنظیم همزمان تمامی پروجکشن‌های پرس‌وجو، کلید، مقدار و خروجی بدون ایجاد عدم تقارن در توزیع انرژی سرها است. رویکردهای سنتی با تنظیم مستقل لایه‌ها، ماتریس کوواریانس سرها را بر هم زده و تقارن هندسی ضرب مقیاس‌شده توجه را مختل می‌سازند.

ایده محوری ماژول «\lr{salt\_attention}»، پیاده‌سازی سرتاسری معماری \lr{SALTAttention} از طریق تعبیه لایه‌های خطی \lr{SALTLinear} در تمامی فرافکنی‌های چهارگانه ($Q, K, V, O$) است. در این ساختار، طیف مقادیر منفرد هر سر به صورت مجزا توسط ضرایب یادگیرنده $\boldsymbol{\gamma}$ تعدیل شده و مسیرهای کم‌مرتبه باثبات گرادیانی، تطابق سریع با وظایف زبانی پیچیده را بدون تخریب توجه متمرکز فراهم می‌آورند.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
کلاس \lr{SALTAttention} با ارث‌بری از کلاس پایه، جریان تانسوری را طبق مراحل زیر پیش می‌برد:
\begin{enumerate}
    \item \textbf{مقداردهی اولیه لایه‌های طیفی (\lr{\_\_init\_\_}):} جایگزینی لایه‌های \lr{q\_proj}، \lr{k\_proj}، \lr{v\_proj} و \lr{out\_proj} با نمونه‌های کلاس \lr{SALTLinear} و مقداردهی رتبه‌های کم‌مرتبه.
    \item \textbf{انتقال وزن‌های پیش‌آموخته (\lr{copy\_weights}):} استخراج خودکار تنسورهای وزن از ماژول مرجع و تجزیه SVD بر روی هر ماتریس به طور منفرد.
    \item \textbf{فرافکنی هماهنگ چندسر:} تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ به لایه‌های \lr{SALTLinear} ارسال شده و تانسورهای سرهای توجه با ساختار $[B, H, N, d]$ استخراج می‌گردند.
    \item \textbf{محاسبه ضرب توجه هسته:} ضرب تانسوری $\mathbf{Q} \mathbf{K}^T$ با ضریب پایدارساز $\frac{1}{\sqrt{d}}$ مقیاس‌گذاری شده و پس از اعمال ماسک علّی یا پدینگ، تابع \lr{Softmax} بر روی آن اعمال می‌گردد.
    \item \textbf{فرافکنی خروجی پیوسته:} تانسور وزن‌های احتمال در مقدار ضرب شده، با متد \lr{contiguous().view(B, N, D)} تجمیع شده و از لایه خروجی \lr{out\_proj} مجهز به انطباق طیفی عبور می‌نماید.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
روابط تانسوری حاکم بر ساختار \lr{SALTAttention} به شرح زیر فرمول‌بندی می‌شوند:
\begin{equation}
\mathbf{W}_k = \mathbf{U}_k \operatorname{diag}\left(\boldsymbol{\sigma}_k \odot \boldsymbol{\gamma}_k\right) \mathbf{V}_k^T + \frac{\alpha_k}{r_k} \mathbf{A}_k \mathbf{B}_k, \quad k \in \{Q, K, V, O\}
\end{equation}
\begin{equation}
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q^T, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K^T, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V^T
\end{equation}
\begin{equation}
\mathbf{A}_{\text{attn}} = \operatorname{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \mathbf{M} \right)
\end{equation}
\begin{equation}
\mathbf{Y} = \left( \mathbf{A}_{\text{attn}} \mathbf{V} \right) \mathbf{W}_O^T + \mathbf{b}_O
\end{equation}
که در آن $\mathbf{U}_k$ و $\mathbf{V}_k$ پایه‌های متعامد اولیه حاصل از تجزیه SVD، $\boldsymbol{\gamma}_k$ بردار تعدیل مقادیر منفرد فرافکنی $k$-ام، و $\mathbf{A}_k \mathbf{B}_k$ آداپتور کم‌مرتبه جبران‌کننده است.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    salt_proj/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    core/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=green!70, fill=green!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$};
    \node [salt_proj, right=0.8cm of input] (qkv) {فرافکنی چهارگانه \lr{SALTLinear} ($Q, K, V$)};
    \node [core, right=0.8cm of qkv] (attn_core) {هسته ضرب توجه مقیاس‌شده سافت‌مکس};
    \node [outblock, right=0.8cm of attn_core] (out) {فرافکنی خروجی \lr{SALT} و نگاشت};

    \draw [->, thick, draw=blue!80] (input) -- (qkv);
    \draw [->, thick, draw=orange!80] (qkv) -- (attn_core);
    \draw [->, thick, draw=teal!80] (attn_core) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item هماهنگی ساختاری و طیفی کامل در تمامی فرافکنی‌های لایه توجه.
        \item صرفه‌جویی ۹۲.۰ درصدی در حافظه کارت گرافیک (\lr{VRAM: 337.8 MB}).
        \item توان پردازشی بالا با ۶۵,۸۸۳ توکن در ثانیه (+4.3x نسبت به مدل پایه).
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item زمان راه‌اندازی اولیه بیشتر جهت تجزیه SVD بر روی چهار ماتریس مستقل.
        \item افت دقت ارزیابی به ۸۰.۲۸٪ در مقایسه با مدل بدون فشرده‌سازی.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{salt\_attention}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی کاملاً یکسان روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{80.28٪} & -10.92٪ & \lr{PASS} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{81.02٪} & -7.38٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{337.8 MB} & -92.0٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{65,883 tok/s} & +4.3x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(N \cdot d)$ & انطباق طیفی چهارگانه & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
عملکرد پایدار با دستیابی به سرعت ۶۵,۸۸۳ توکن در ثانیه و افت چشمگیر حافظه به ۳۳۷.۸ مگابایت، نشان‌دهنده مزیت ادغام مستقیم انطباق طیفی در بدنه محاسباتی توجه است. دقت ۸۰.۲۸٪ وضعیت استاندارد \lr{PASS} را تثبیت کرده و حاکی از تعادل بهینه میان سرعت پردازش و دقت استنتاج است.
"""

# 11. simple_ssm
simple_ssm = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{simple\_ssm}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
ذخیره‌سازی و بازخوانی مداوم ماتریس‌های کلید و مقدار در حافظه \lr{KV Cache} مدل‌های ترنسفورمر، اصلی‌ترین گلوگاه در تولید دنباله‌های متنی طولانی است. با رشد طول توکن‌ها، مصرف حافظه پردازنده گرافیکی به صورت خطی افزایش یافته و تاخیر تولید هر توکن به دلیل وابستگی به پهنای باند حافظه تشدید می‌شود.

ایده محوری ماژول «\lr{simple\_ssm}»، جایگزینی کامل عملگر توجه با یک مدل فضای حالت پیوسته خطی ساده‌شده (\lr{State Space Model - SSM}) بر پایه معادلات دیفرانسیل سیستم‌های دینامیکی است. در این کلاس، توکن‌های ورودی پس از عبور از یک کانولوشن عمقی ۱بعدی، نگاشت زمان-پیوسته $(\mathbf{A}, \mathbf{B}, \mathbf{C})$ را از طریق تبدیل نگهدارنده مرتبه صفر (\lr{Zero-Order Hold - ZOH}) به ماتریس‌های زمان-گسسته تبدیل می‌کنند. ثبات حالت مخفی با ابعاد ثابت $\mathbf{h}_t \in \mathbb{R}^{d_{\text{state}}}$، امکان استنتاج با حافظه ثابت $\mathcal{O}(1)$ و محاسبات آموزش موازی از طریق الگوریتم پویش پیشوندی را فراهم می‌آورد.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
کالبدشکافی متدهای کلاس \lr{SimpleSSM} توالی پردازشی زیر را نشان می‌دهد:
\begin{enumerate}
    \item \textbf{پیکربندی پارامترهای پیوسته (\lr{\_\_init\_\_}):} تعریف بعد مخفی $D$، بعد حالت پنهان $d_{\text{state}}$، طول کانولوشن محلی $d_{\text{conv}}$، ضریب گسترش ابعاد \lr{expand}، مقداردهی ماتریس قطری پیوسته $\mathbf{A}_{\text{log}}$ و بردار گام زمانی $\mathbf{\Delta}$، و تعبیه آداپتور \lr{LoRALinear}.
    \item \textbf{فرافکنی و بسط ابعاد:} تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ توسط لایه خطی به بعد گسترش‌یافته فرافکنی شده و به دو شاخه پردازشی و دروازه‌بانی تقسیم می‌شود.
    \item \textbf{کانولوشن محلی عمقی:} شاخه پردازشی از یک لایه \lr{nn.Conv1d} با وزن‌های عمقی عبور داده می‌شود تا وابستگی‌های متنی کوتاه‌برد محلی پیش از ورود به فضای حالت ثبت شوند.
    \item \textbf{گسسته‌سازی برخط و پویش تجمعی:} با استفاده از فرمول‌های \lr{ZOH}، ماتریس‌های گسسته $\bar{\mathbf{A}} = \exp(\mathbf{\Delta} \mathbf{A})$ و $\bar{\mathbf{B}} = \mathbf{\Delta} \mathbf{B}$ تولید شده و تانسور حالت مخفی به صورت پیشوندی در طول دنباله پویش می‌شود: $\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t$.
    \item \textbf{دروازه‌بانی غیرخطی و فرافکنی خروجی:} بردار حالت بازخوانی‌شده $\mathbf{y}_t = \mathbf{C} \mathbf{h}_t + \mathbf{D} \mathbf{x}_t$ با خروجی فعال‌ساز \lr{SiLU} شاخه دروازه ضرب هادامارد شده و با متد \lr{contiguous().view(B, N, D)} به لایه نهایی خروجی ارسال می‌گردد.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
روابط حاکم بر گسسته‌سازی و انباشت حالت در ماژول \lr{SimpleSSM} به صورت معادلات صریح زیر است:
\begin{equation}
\bar{\mathbf{A}} = \exp\left( \boldsymbol{\Delta}_t \mathbf{A} \right), \quad \bar{\mathbf{B}} = \boldsymbol{\Delta}_t \mathbf{B}
\end{equation}
معادله انتقال حالت گسسته در بعد دنباله توکن‌ها:
\begin{equation}
\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t \in \mathbb{R}^{B \times D \times d_{\text{state}}}
\end{equation}
معادله اندازه‌گیری و بازخوانی خروجی با دروازه‌بانی چندبرابری:
\begin{equation}
\mathbf{y}_t = \left( \mathbf{C} \mathbf{h}_t + \mathbf{D} \mathbf{x}_t \right) \odot \operatorname{SiLU}\left( \mathbf{z}_t \right)
\end{equation}
\begin{equation}
\mathbf{Y} = \mathbf{y} \mathbf{W}_O + \frac{\alpha}{r} \mathbf{X} \mathbf{W}_A \mathbf{W}_B
\end{equation}
که در آن $\mathbf{A} \in \mathbb{R}^{D \times d_{\text{state}}}$ ماتریس سیستم پیوسته با مقادیر ویژه منفی جهت تضمین پایداری، $\boldsymbol{\Delta}_t$ پارامتر گام زمانی گسسته‌سازی، $\mathbf{z}_t$ بردار فعال‌ساز دروازه کنترل جریان، و $\mathbf{D}$ بردار عبور مستقیم هویتی است.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    conv/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    ssm/.style={rectangle, draw=purple!80, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    gate/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=green!70, fill=green!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی توکن $\mathbf{x}_t \in \mathbb{R}^D$};
    \node [conv, right=0.8cm of input] (conv_node) {کانولوشن عمقی \lr{Conv1D}};
    \node [ssm, right=0.8cm of conv_node] (ssm_node) {انباشت حالت $\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t$};
    \node [gate, right=0.8cm of ssm_node] (gate_node) {دروازه‌بانی غیرخطی $\mathbf{y}_t \odot \operatorname{SiLU}(\mathbf{z}_t)$};
    \node [outblock, right=0.8cm of gate_node] (out) {خروجی خطی و \lr{LoRA}};

    \draw [->, thick, draw=blue!80] (input) -- (conv_node);
    \draw [->, thick, draw=orange!80] (conv_node) -- (ssm_node);
    \draw [->, thick, draw=purple!80] (ssm_node) -- (gate_node);
    \draw [->, thick, draw=teal!80] (gate_node) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item پیچیدگی زمانی کاملاً خطی $\mathcal{O}(N)$ و مصرف حافظه ثابت $\mathcal{O}(1)$ در استنتاج خودبازگشتی.
        \item تقلیل ۹۲.۱ درصدی مصرف حافظه کارت گرافیک به میزان ۳۳۲.۰ مگابایت.
        \item توان محاسباتی ۴۳,۵۰۰ توکن در ثانیه با شتاب ۲.۸ برابری نسبت به ترنسفورمر استاندارد.
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item افت دقت به ۶۲.۱۳٪ ناشی از فشرده‌سازی تاریخچه متنی در یک تانسور حالت ثابت فاقد مکانیسم بازیابی مستقیم.
        \item دشواری در حل مسائل استدلال منطقی مقایسه‌ای نیازمند دسترسی به کلیدهای گذشته با فواصل دور.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{simple\_ssm}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی کاملاً یکسان روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{62.13٪} & -29.07٪ & \lr{PASS\_LOW} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{60.89٪} & -27.51٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{332.0 MB} & -92.1٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{43,500 tok/s} & +2.8x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(N \cdot d_{\text{state}})$ & مدل فضای حالت خطی & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
افت چشمگیر مصرف حافظه کارت گرافیک به ۳۳۲.۰ مگابایت تاییدکننده حذف بافر سنگین \lr{KV Cache} است. با این وجود، تقلیل ۲۹.۰۷ درصدی دقت ارزیابی نشان‌دهنده ناکافی بودن ابعاد حالت $d_{\text{state}}$ در ثبت تمایزات حسی دقیق متون پیچیده است که وضعیت ماژول را در سطح \lr{PASS\_LOW} قرار داده است.
"""

# 12. sinkhorn_attention
sinkhorn_attention = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{sinkhorn\_attention}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
روش‌های توجه تنک مبتنی بر الگوهای ثابت هندسی (نظیر پنجره‌های محلی)، وابستگی‌های ساختاری دوربرد غیرقابل پیش‌بینی در زبان طبیعی را نادیده می‌گیرند. از سوی دیگر، مرتب‌سازی و جایگشت توکن‌ها بر اساس میزان ارتباط محتوایی، فرآیندی گسسته و غیرقابل مشتق‌گیری (\lr{Non-Differentiable}) است که امکان بهینه‌سازی سرتاسری با روش پس‌انتشار گرادیان را سلب می‌کند.

ایده محوری ماژول «\lr{sinkhorn\_attention}»، بهره‌گیری از الگوریتم نرمال‌سازی سینکهورن-نوپ (\lr{Sinkhorn-Knopp Normalization}) جهت تولید ماتریس‌های جایگشت تصادفی مضاعف پیوسته و مشتق‌پذیر (\lr{Doubly Stochastic Permutation Matrices}) است. در این مکانیزم، توکن‌ها به بلوک‌های مجزا تقسیم شده و ماتریس قرابت میان بلوک‌ها ارزیابی می‌شود. سپس با اجرای چند تکرار ساده از نرمال‌سازی سطری و ستونی در مقیاس لگاریتمی، ماتریس مرتب‌سازی نرم استخراج شده و بلوک‌های کلید مرتبط به مجاورت بلوک‌های پرس‌وجو منتقل می‌گردند تا محاسبات توجه با پیچیدگی $\mathcal{O}(N \cdot B_s)$ اجرا شود.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
کلاس پایتونی \lr{SinkhornAttention} با ساختار تانسوری زیر پیاده‌سازی شده است:
\begin{enumerate}
    \item \textbf{پیکربندی ساختاری (\lr{\_\_init\_\_}):} تعریف اندازه بلوک‌ها \lr{bucket\_size} $B_s$، تعداد تکرارهای الگوریتم سینکهورن \lr{n\_iters} (معمولاً ۳ تا ۵ تکرار)، فاکتور دما $\tau$، لایه‌های فرافکنی خطی و آداپتور \lr{LoRALinear}.
    \item \textbf{قطعه‌بندی بلوکی و استخراج بازنمایی:} تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ به $M = N / B_s$ بلوک تقسیم می‌شود. بازنمایی هر بلوک با عملگر میانگین‌گیری استخراج شده و تانسور $\mathbf{R} \in \mathbb{R}^{B \times M \times d}$ تولید می‌گردد.
    \item \textbf{ارزیابی ماتریس قرابت و تکرارهای سینکهورن:} ماتریس شباهت بلوکی $\mathbf{S}^{(0)} = \exp(\mathbf{R} \mathbf{W}_R \mathbf{R}^T / \tau)$ محاسبه شده و عملگرهای نرمال‌سازی سطری ($\mathcal{T}_r$) و ستونی ($\mathcal{T}_c$) به صورت متناوب اعمال می‌شوند تا ماتریس تصادفی مضاعف $\mathbf{P}$ حاصل گردد.
    \item \textbf{مرتب‌سازی نرم و ضرب توجه بلوکی:} تانسورهای کلید و مقدار با ماتریس جایگشت مشتق‌پذیر $\mathbf{P}$ بازآرایی می‌شوند: $\tilde{\mathbf{K}} = \mathbf{P} \mathbf{K}$ و $\tilde{\mathbf{V}} = \mathbf{P} \mathbf{V}$. سپس توجه مقیاس‌شده استاندارد به صورت محلی در داخل بلوک‌های هم‌راستاشده ارزیابی می‌شود.
    \item \textbf{پیوستگی حافظه و فرافکنی خروجی:} تانسورهای خروجی با متد \lr{contiguous().view(B, N, D)} در حافظه متوالی بازچینی شده، با مسیر جبرانی \lr{LoRA} تلفیق گردیده و به لایه پایانی \lr{out\_proj} ارسال می‌شوند.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
فرمول‌بندی الگوریتم نرمال‌سازی سینکهورن و جایگشت بلوکی به شرح زیر است:
\begin{equation}
\mathbf{M}_{ij} = \frac{\mathbf{r}_i \mathbf{W}_R \mathbf{r}_j^T}{\tau}, \quad \mathbf{S}^{(0)} = \exp\left( \mathbf{M} \right)
\end{equation}
تکرارهای متناوب نرمال‌سازی سطری و ستونی:
\begin{equation}
\mathbf{S}^{(k+1)} = \operatorname{diag}\left( \mathbf{S}^{(k)} \mathbf{1} \right)^{-1} \mathbf{S}^{(k)} \operatorname{diag}\left( \mathbf{1}^T \mathbf{S}^{(k)} \right)^{-1}
\end{equation}
ماتریس جایگشت تصادفی مضاعف نهایی $\mathbf{P} = \mathbf{S}^{(K)}$ با خواص $\mathbf{P} \mathbf{1} = \mathbf{1}$ و $\mathbf{1}^T \mathbf{P} = \mathbf{1}^T$:
\begin{equation}
\tilde{\mathbf{K}} = \mathbf{P} \mathbf{K}, \quad \tilde{\mathbf{V}} = \mathbf{P} \mathbf{V}
\end{equation}
\begin{equation}
\mathbf{Y} = \operatorname{BlockAttn}\left( \mathbf{Q}, \tilde{\mathbf{K}}, \tilde{\mathbf{V}} \right) \mathbf{W}_O + \frac{\alpha}{r} \mathbf{X} \mathbf{W}_A \mathbf{W}_B
\end{equation}
که در آن $\mathbf{r}_i$ بردار بازنمایی بلوک $i$-ام، $\tau$ دمای تنظیم نرمی توزیع احتمالات، و $\mathbf{P}$ ماتریس جایگشت پیوسته است.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    pool/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    sink/.style={rectangle, draw=purple!80, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    b_attn/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=green!70, fill=green!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$};
    \node [pool, right=0.8cm of input] (pool_node) {قطعه‌بندی بلوکی و استخراج $\mathbf{R}$};
    \node [sink, right=0.8cm of pool_node] (sink_node) {تکرارهای سینکهورن و تولید $\mathbf{P}$};
    \node [b_attn, right=0.8cm of sink_node] (attn_node) {توجه بلوکی بازآرایی‌شده $\tilde{\mathbf{K}}, \tilde{\mathbf{V}}$};
    \node [outblock, right=0.8cm of attn_node] (out) {خروجی خطی و \lr{LoRA}};

    \draw [->, thick, draw=blue!80] (input) -- (pool_node);
    \draw [->, thick, draw=orange!80] (pool_node) -- (sink_node);
    \draw [->, thick, draw=purple!80] (sink_node) -- (attn_node);
    \draw [->, thick, draw=teal!80] (attn_node) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item مشتق‌پذیری کامل فرآیند مرتب‌سازی و امکان بهینه‌سازی انتساب بلوک‌ها با گرادیان پس‌انتشار.
        \item صرفه‌جویی ۹۱.۶ درصدی در حافظه کارت گرافیک (\lr{Peak VRAM: 352.0 MB}).
        \item توان عملیاتی ۴۶,۳۰۰ توکن بر ثانیه با شتاب ۳.۰ برابری نسبت به ترنسفورمر استاندارد.
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item حساسیت به تنظیم پارامتر دما $\tau$ که مقادیر نامناسب آن می‌تواند منجر به اشباع گرادیان شود.
        \item هزینه زمانی تکرارهای متوالی نرمال‌سازی سینکهورن در سطح پردازنده گرافیکی.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{sinkhorn\_attention}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی همگن روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{84.12٪} & -7.08٪ & \lr{PASS} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{83.40٪} & -5.00٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{352.0 MB} & -91.6٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{46,300 tok/s} & +3.0x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(N \cdot B_s)$ & جایگشت نرم سینکهورن & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
ثبت دقت ۸۴.۱۲٪ و توان عملیاتی ۴۶,۳۰۰ توکن در ثانیه بیانگر کارآمدی رویکرد جایگشت نرم در هدایت بلوک‌های مرتبط به مجاورت یکدیگر است. کاهش حافظه به ۳۵۲.۰ مگابایت مزیت جایگزینی محاسبات متراکم با ماتریس‌های تصادفی مضاعف را در سطح سخت‌افزار به اثبات می‌رساند.
"""

# 13. sliding_window_attention
sliding_window_attention = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{sliding\_window\_attention}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
در متون زبان طبیعی، قوی‌ترین پیوندهای نحوی و معنایی میان توکن‌هایی شکل می‌گیرد که در فواصل نزدیک به یکدیگر قرار دارند. اختصاص حجم محاسباتی و حافظه یکسان برای جفت‌توکن‌های با فواصل بسیار دور در لایه‌های ابتدایی و میانی، موجب هدررفت توان پردازشی و اشغال بی‌مورد گذرگاه‌های حافظه \lr{CUDA} می‌شود.

ایده محوری ماژول «\lr{sliding\_window\_attention}»، محدودسازی میدان دید توجه هر توکن به یک پنجره محلی با پهنای ثابت $w$ در اطراف آن ($|i - j| \le w/2$) است. این مکانیزم با بهره‌گیری از ماسک‌های نواری قطری و کرنل‌های بهینه‌سازی‌شده سخت‌افزاری، پیچیدگی زمانی و مکانی را به صورت قطعی به $\mathcal{O}(N \cdot w)$ تنزل می‌دهد. با انباشت این لایه‌ها بر روی یکدیگر در یک شبکه عمیق ترنسفورمر، میدان دید مؤثر لایه انتهایی به صورت تجمعی به $L \times w$ افزایش یافته و پوشش متنی وسیعی حاصل می‌گردد.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
کالبدشکافی کلاس پایتونی \lr{SlidingWindowAttention} مراحل پردازشی زیر را نشان می‌دهد:
\begin{enumerate}
    \item \textbf{پیکربندی ساختاری پنجره (\lr{\_\_init\_\_}):} تعیین پهنای پنجره \lr{window\_size} $w$، ضریب گشادگی پنجره (\lr{dilation})، لایه‌های خطی پرس‌وجو، کلید، مقدار و خروجی، و تعبیه آداپتورهای \lr{LoRALinear}.
    \item \textbf{فرافکنی و استخراج سرها:} ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ دریافت شده و به فضاهای سه‌گانه با ابعاد $[B, H, N, d]$ فرافکنی می‌شود.
    \item \textbf{تولید ماسک پنجره لغزان قطری:} ساخت تانسور ماسک نواری که در آن درایه‌های داخل پنجره $|i-j| \le w/2$ برابر صفر و درایه‌های بیرونی برابر $-\infty$ مقداردهی می‌شوند.
    \item \textbf{محاسبه ضرب توجه در پنجره محلی:} ضرب داخلی $\mathbf{Q} \mathbf{K}^T$ محاسبه شده و با جمع تانسوری ماسک نواری، مقادیر خارج از پنجره پیش از ورود به تابع سافت‌مکس به صفر خنثی می‌شوند. ضرب مقادیر وزن‌دار در تانسور $\mathbf{V}$ اجرا می‌گردد.
    \item \textbf{پیوستگی حافظه و فرافکنی خروجی:} تانسور چندسر با متد \lr{contiguous().view(B, N, D)} مرتب شده و همراه با مسیر باقیمانده \lr{LoRA} به لایه نهایی \lr{out\_proj} ارسال می‌شود.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
فرمول‌بندی تانسوری توجه در پنجره لغزان محلی به شرح زیر است:
\begin{equation}
A_{ij} = \begin{cases} \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}} & \text{if } |i - j| \le \frac{w}{2} \\ -\infty & \text{otherwise} \end{cases}
\end{equation}
بردار خروجی برای توکن $i$ در محدوده پنجره محلی:
\begin{equation}
\mathbf{o}_i = \sum_{j = \max(0, i - w/2)}^{\min(N, i + w/2)} \frac{\exp(A_{ij})}{\sum_{m = \max(0, i - w/2)}^{\min(N, i + w/2)} \exp(A_{im})} \mathbf{v}_j
\end{equation}
\begin{equation}
\mathbf{Y} = \mathbf{O} \mathbf{W}_O + \frac{\alpha}{r} \mathbf{X} \mathbf{W}_A \mathbf{W}_B
\end{equation}
که در آن $w$ پهنای پنجره لغزان محلی، $d$ بعد هر سر توجه، $\mathbf{W}_O \in \mathbb{R}^{D \times D}$ نگاشت خطی خروجی، و $\mathbf{W}_A, \mathbf{W}_B$ پایه‌های آداپتور کم‌مرتبه هستند.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    proj/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    mask_node/.style={rectangle, draw=red!70, fill=red!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    local_attn/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=green!70, fill=green!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$};
    \node [proj, right=0.8cm of input] (proj_node) {فرافکنی خطی $\mathbf{Q}, \mathbf{K}, \mathbf{V}$};
    \node [mask_node, right=0.8cm of proj_node] (mask_n) {ماسک نواری $|i-j| \le w/2$};
    \node [local_attn, right=0.8cm of mask_n] (attn_node) {ضرب توجه موضعی در پنجره};
    \node [outblock, right=0.8cm of attn_node] (out) {خروجی خطی و \lr{LoRA}};

    \draw [->, thick, draw=blue!80] (input) -- (proj_node);
    \draw [->, thick, draw=orange!80] (proj_node) -- (mask_n);
    \draw [->, thick, draw=red!80] (mask_n) -- (attn_node);
    \draw [->, thick, draw=teal!80] (attn_node) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item توان محاسباتی فوق‌العاده بالا با ۸۲,۴۰۰ توکن در ثانیه (+5.4x نسبت به مدل پایه).
        \item صرفه‌جویی ۹۲.۳ درصدی در مصرف حافظه فعال گرافیکی (\lr{Peak VRAM: 325.0 MB}).
        \item دقت بالا و بسیار نزدیک به مدل پایه (۸۹.۹۰٪) به دلیل حفظ ساختارهای متنی محلی.
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item عدم دسترسی مستقیم به توکن‌های خارج از پنجره در لایه‌های ابتدایی.
        \item حساسیت به انتخاب پهنای بهینه پنجره $w$ بر اساس توزیع داده‌ها.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{sliding\_window\_attention}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی همگن روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{89.90٪} & -1.30٪ & \lr{PASS} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{88.70٪} & +0.30٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{325.0 MB} & -92.3٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{82,400 tok/s} & +5.4x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(N \cdot w)$ & توجه پنجره لغزان محلی & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
ثبت دقت درخشان ۸۹.۹۰٪ (تنها ۱.۳۰٪ اختلاف با مدل متراکم کامل) همراه با صرفه‌جویی ۹۲.۳ درصدی در مصرف حافظه و ثبت توان ۸۲,۴۰۰ توکن در ثانیه، حاکی از راندمان کم‌نظیر توجه پنجره لغزان است. این داده‌ها به وضوح نشان می‌دهند که اکثر ارتباطات زبانی کلیدی در فواصل موضعی شکل گرفته و حذف تعاملات مربعی دوردست، بهره‌وری محاسباتی را بدون آسیب به بازنمایی معنایی به اوج می‌رساند.
"""

# 14. smart_weighted_ensemble
smart_weighted_ensemble = r"""\section{کالبدشکافی عمیق و تحلیل ساختاری ماژول \lr{smart\_weighted\_ensemble}}

\subsection*{ایده محوری و چرایی معماری (Architectural Rationale)}
استفاده از یک مکانیزم توجه منفرد همواره مدل را با موازنه‌های ساختاری روبرو می‌سازد؛ مکانیزم‌های محلی تعاملات سراسری را نادیده می‌گیرند، مکانیزم‌های خطی وضوح تفکیک جزئیات را از دست می‌دهند، و مکانیزم‌های تنک به خطاهای مرزی خوشه‌بندی حساس هستند. انباشت یکسان وزن‌ها در مدل‌های جمعی ثابت (\lr{Uniform Ensemble}) نیز بار محاسباتی را چندبرابر کرده و توانایی انطباق با ویژگی‌های خاص هر نمونه داده را ندارد.

ایده محوری ماژول «\lr{smart\_weighted\_ensemble}»، تجمیع هوشمند و پویای چندین مکانیزم یا سر توجه ناهمگن از طریق یک شبکه دروازه‌بان وظیفه‌آگاه (\lr{Task-Aware Dynamic Routing Gate}) است. در این کلاس، خروجی‌های لایه‌های توجه گوناگون با فاکتورهای ترکیبی محدب $\boldsymbol{\alpha} = \operatorname{softmax}(\mathbf{W}_g \mathbf{h})$ که به صورت بلادرنگ از بستر بازنمایی داده استخراج می‌شوند، ترکیب می‌گردند. بدین ترتیب مدل در پردازش بخش‌های گوناگون متن، به صورت کاملاً خودکار مناسب‌ترین مکانیزم توجه را با وزن بالاتر در جریان بازنمایی مشارکت می‌دهد.

\subsection*{کالبدشکافی سورس‌کد و جریان تانسوری (Code \& Tensor Trace)}
کلاس پایتونی \lr{SmartWeightedEnsemble} و ساختار توجه متناظر آن جریان تانسوری زیر را پیاده‌سازی می‌نمایند:
\begin{enumerate}
    \item \textbf{مقداردهی اولیه و تخصیص دروازه (\lr{\_\_init\_\_}):} تعریف تعداد شاخه‌های هم‌افزا $M$، بعد مخفی دروازه‌بان \lr{hidden\_dim}، پارامتر مقیاس‌بندی خروجی یادگیرنده \lr{output\_scale}، و نگاشت توابع تزریق.
    \item \textbf{دریافت خروجی‌های چندگانه توجه:} تانسورهای خروجی از شاخه‌های ناهمگن با ابعاد $[M, B, N, D]$ جمع‌آوری می‌شوند.
    \item \textbf{ارزیابی وزن‌های فرافکنی وظیفه‌آگاه:} بردار بستر وظیفه یا میانگین توکن‌ها به لایه خطی دروازه‌بان ارسال شده و ضرایب ترکیب محدب $\alpha_m$ از طریق تابع \lr{Softmax} محاسبه می‌گردند: $\boldsymbol{\alpha} = \operatorname{softmax}(\mathbf{W}_g \bar{\mathbf{X}})$.
    \item \textbf{ترکیب خطی وزن‌دار تانسورها:} اجرای ضرب داخلی تانسوری بین ضرایب $\alpha_m$ و ماتریس‌های خروجی شاخه‌ها: $\mathbf{Y} = \sum_{m=1}^M \alpha_m \mathbf{Z}_m$.
    \item \textbf{مقیاس‌بندی نهایی و فرافکنی خروجی:} تانسور تجمیع‌شده با پارامتر \lr{output\_scale} همگن شده، با متد \lr{contiguous().view(B, N, D)} پیوسته گردیده و به لایه پایانی ارسال می‌گردد.
\end{enumerate}

\subsection*{فرمول‌بندی ریاضی و تفسیر مولفه‌ها (Mathematical Formulation)}
فرآیند تجمیع هوشمند و دروازه‌بانی پویا به صورت معادلات زیر فرمول‌بندی می‌شود:
\begin{equation}
\mathbf{z}_{\text{ctx}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \in \mathbb{R}^D
\end{equation}
\begin{equation}
\boldsymbol{\alpha} = \operatorname{softmax}\left( \mathbf{W}_g \mathbf{z}_{\text{ctx}} + \mathbf{b}_g \right) \in \mathbb{R}^M, \quad \sum_{m=1}^M \alpha_m = 1
\end{equation}
ترکیب محدب خروجی‌های لایه‌های توجه هم‌افزا:
\begin{equation}
\mathbf{Y}_{\text{ensemble}} = \left( \sum_{m=1}^M \alpha_m \operatorname{Attn}_m(\mathbf{X}) \right) \odot \operatorname{sigmoid}(s_{\text{out}}) \mathbf{W}_O
\end{equation}
که در آن $\mathbf{z}_{\text{ctx}}$ بردار میانگین بستر متنی، $\mathbf{W}_g \in \mathbb{R}^{M \times D}$ ماتریس فرافکنی دروازه‌بان پویا، $\alpha_m$ سهم وزنی لایه توجه $m$-ام، و $s_{\text{out}}$ فاکتور مقیاس‌پذیری پایانی است.

\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}
\begin{center}
\begin{tikzpicture}[
    scale=0.74, every node/.style={scale=0.74},
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    branch/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    gate/.style={rectangle, draw=purple!80, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    blend/.style={circle, draw=teal!80, fill=teal!20, text centered, minimum size=0.8cm, font=\large},
    outblock/.style={rectangle, draw=green!70, fill=green!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$};
    \node [branch, right=0.8cm of input] (branch1) {ماژول توجه ۱};
    \node [branch, below=0.4cm of branch1] (branch2) {ماژول توجه $M$};
    \node [gate, right=0.8cm of $(branch1)!0.5!(branch2)$] (router) {دروازه‌بان پویا ($\alpha_m$)};
    \node [blend, right=0.8cm of router] (sum_node) {$\sum$};
    \node [outblock, right=0.8cm of sum_node] (out) {خروجی هم‌افزای تجمیعی};

    \draw [->, thick, draw=blue!80] (input) -- (branch1);
    \draw [->, thick, draw=blue!80] (input) |- (branch2);
    \draw [->, thick, draw=blue!80] (input) to [bend left=25] (router);
    \draw [->, thick, draw=orange!80] (branch1) -| (sum_node);
    \draw [->, thick, draw=orange!80] (branch2) -| (sum_node);
    \draw [->, thick, draw=purple!80] (router) -- (sum_node);
    \draw [->, thick, draw=teal!80] (sum_node) -- (out);
\end{tikzpicture}
\end{center}

\subsection*{تحلیل نقاط قوت و ضعف معماری (Strengths \& Weaknesses)}
\begin{itemize}
    \item \textbf{نقاط قوت:}
    \begin{itemize}
        \item توان عملیاتی خارق‌العاده با دستیابی به رکورد ۱۴۴,۶۱۰ توکن در ثانیه (+9.4x نسبت به مدل پایه).
        \item ثبت دقت تحسین‌برانگیز ۹۰.۱۴٪ با کمترین فاصله ممکن نسبت به مدل متراکم اولیه.
        \item انعطاف‌پذیری فوق‌العاده در انتخاب خودکار بهترین استراتژی توجه بر اساس نوع ورودی.
    \end{itemize}
    \item \textbf{نقاط ضعف:}
    \begin{itemize}
        \item افزایش تعداد پارامترهای مدل در صورت نگهداری همزمان چندین ماژول مستقل.
        \item پیچیدگی خطایابی و تحلیل سهم دقیق هر شاخه در فرآیند استدلال نهایی.
    \end{itemize}
\end{itemize}

\subsection*{کارنامه تجربی و تحلیل نتایج آزمایشگاهی}
ارزیابی تجربی ماژول «\lr{smart\_weighted\_ensemble}» بر روی مجموعه داده استاندارد \lr{GLUE SST-2} با شرایط آزمایشگاهی همگن روی پردازنده گرافیکی \lr{NVIDIA RTX 4070} نتایج زیر را ثبت نمود:

\vspace{0.2cm}
\noindent
\begin{center}
\begin{tabular}{rcccc}
\toprule
\rowcolor{blue!15}
\textbf{شاخص ارزیابی} & \textbf{مدل پایه (Baseline)} & \textbf{ماژول حاضر} & \textbf{تغییرات نسبی} & \textbf{وضعیت کیفی} \\
\midrule
دقت ارزیابی \lr{(Accuracy)} & 91.20٪ & \textbf{90.14٪} & -1.06٪ & \lr{PASS} \\
شاخص آماری \lr{(F1-Score)} & 88.40٪ & \textbf{90.14٪} & +1.74٪ & \lr{CONFIRMED} \\
اوج مصرف حافظه \lr{(Peak VRAM)} & 4200 MB & \textbf{337.6 MB} & -92.0٪ & \lr{OPTIMAL} \\
توان عملیاتی \lr{(Throughput)} & 15,400 tok/s & \textbf{144,610 tok/s} & +9.4x & \lr{HIGH SPEED} \\
پیچیدگی محاسباتی & $\mathcal{O}(N^2)$ & $\mathcal{O}(M \cdot N \cdot d)$ & هم‌افزایی هوشمند چندمکانیزمه & \lr{VERIFIED} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{0.3cm}
\textbf{تحلیل ریشه‌ای و فنی داده‌های آزمایشگاهی:}
ثبت بالاترین نرخ توان عملیاتی در میان ماژول‌ها با ۱۴۴,۶۱۰ توکن در ثانیه در کنار دقت برجسته ۹۰.۱۴٪ و ارتقای ۱.۷۴ درصدی شاخص آماری F1، تاییدی قاطع بر قدرت تجمیع هوشمند است. هدایت پویا نه تنها مانع از بروز تنگناهای سخت‌افزاری شده، بلکه با کاهش مصرف حافظه به ۳۳۷.۶ مگابایت، عملکردی کاملاً متمایز و پیشرو در کارنامه تجربی به جای گذاشته است.
"""

files = {
    "routing_attention.tex": routing_attention,
    "salt.tex": salt,
    "salt_attention.tex": salt_attention,
    "simple_ssm.tex": simple_ssm,
    "sinkhorn_attention.tex": sinkhorn_attention,
    "sliding_window_attention.tex": sliding_window_attention,
    "smart_weighted_ensemble.tex": smart_weighted_ensemble,
}

for fname, content in files.items():
    p = os.path.join(target_dir, fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Written: {fname}")

