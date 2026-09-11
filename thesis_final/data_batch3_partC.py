# -*- coding: utf-8 -*-
"""Data definitions for Batch 3: modules 91 to 101."""

PART_C = {
    "routing_attention": {
        "title": "Routing Attention: Spherical K-Means Clustered Attention",
        "philosophy": r"معماری \lr{routing\_attention} توکن‌ها را در یک فضای برداری کروی خوشه‌بندی می‌کند. هر پرس‌وجو به جای جستجو در کل دنباله، صرفاً به توکن‌های کلیدی که در همان سلول ورونوی کروی قرار گرفته‌اند توجه می‌کند تا هزینه محاسباتی از حالت مربعی به شبه‌خطی دگرگون گردد.",
        "analogy": r"سازماندهی مقالات در کنفرانس‌های علمی بین‌المللی با ۱۰ هزار مقاله ارسالی: مقالات بر اساس شباهت موضوعی به ۲۰ پنل تخصصی (سلول‌های ورونوی کروی) هدایت می‌شوند و داوران هر پنل فقط مقالات حوزه تخصصی خود را ارزیابی می‌کنند، به جای آنکه هر داور مجبور باشد تمام ۱۰ هزار مقاله ارسالی را مطالعه کند.",
        "derivation": r"""تانسورهای کلید با الگوریتم کروی کی-میانگین (\lr{Spherical k-means}) به $K$ خوشه با مراکز متعامد $\boldsymbol{\mu}_k$ تقسیم می‌شوند:
\begin{equation}
c(\mathbf{x}_i) = \arg\max_{k \in \{1, \dots, K\}} \frac{\langle \mathbf{x}_i, \boldsymbol{\mu}_k \rangle}{\|\mathbf{x}_i\|_2 \|\boldsymbol{\mu}_k\|_2}
\end{equation}
توجه صرفاً درون خوشه‌های هم‌نام محاسبه می‌شود:
\begin{equation}
\operatorname{RoutingAttn}(\mathbf{q}_i) = \sum_{j: c(\mathbf{k}_j) = c(\mathbf{q}_i)} \frac{\exp(\mathbf{q}_i \mathbf{k}_j^\top / \sqrt{d})}{\sum_{l: c(\mathbf{k}_l) = c(\mathbf{q}_i)} \exp(\mathbf{q}_i \mathbf{k}_l^\top / \sqrt{d})} \mathbf{v}_j
\end{equation}
پیچیدگی کل فرآیند به $\mathcal{O}(N \cdot \frac{N}{K} d)$ کاهش می‌یابد.""",
        "tensor_trace": [
            r"تصویرسازی بردارهای پرس‌وجو و کلید بر روی کره واحد اقلیدسی با نرمال‌سازی $L_2$.",
            r"تعیین نزدیک‌ترین بردار مرکزی با ضرب ماتریسی در ماتریس مراکز خوشه‌ها.",
            r"مرتب‌سازی توکن‌ها بر مبنای برچسب خوشه با عملگرهای ایندکس‌دهی موازی.",
            r"محاسبه ضرب توجه بلوکی متراکم درون هر خوشه به صورت مستقل.",
            r"بازنشانی چینش خروجی به ترتیب زمانی اولیه با شاخص‌های معکوس."
        ],
        "roofline": [
            r"\textbf{سازگاری با کش سریع پردازنده:} خوشه‌ها ابعادی کوچک دارند و ماتریس توجه هر خوشه درون \lr{SRAM} جا می‌گیرد.",
            r"\textbf{حذف ۹۰ درصدی تراکنش‌های زائد:} توکن‌های نامرتبط در خوشه‌های دیگر هرگز از حافظه اصلی فراخوانی نمی‌شوند."
        ],
        "status": "PASS", "acc": "83.40", "f1": "81.2", "vram": "780.0", "thr": "33,100", "vram_pct": "81.4",
        "justification": r"ثبت دقت 83.40 درصد و مصرف رم 780.0 مگابایت (کاهش 81.4 درصدی نسبت به مدل مبنای ۴۲۰۰ مگابایت) همراه با سرعت 33,100 توکن در ثانیه، تأییدکننده همبستگی زاویه‌ای توکن‌ها در فضای معنایی است و نشان می‌دهد خوشه‌بندی کروی انحراف خطای بسیار اندکی تولید می‌کند.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    clus/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [clus, right=1.0cm of input] (norm) {نرمال‌سازی کروی};
    \node [clus, right=1.0cm of norm] (cluster) {خوشه‌بندی $K$-Means};
    \node [outblock, right=1.0cm of cluster] (out) {توجه درون‌خوشه‌ای};

    \draw [->, thick, draw=blue!80] (input) -- (norm);
    \draw [->, thick, draw=orange!80] (norm) -- (cluster);
    \draw [->, thick, draw=orange!80] (cluster) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific visualization of spherical routing attention. Showing hundreds of glowing data points projected onto the surface of a luminous sphere, partitioned into vibrant Voronoi geometric cells with localized laser interconnections, 8k resolution --ar 16:9"
    },

    "rpc": {
        "title": "RPC: Random Projection Compression",
        "philosophy": r"مکانیزم \lr{rpc} (فشرده‌سازی با پروجکشن تصادفی) از لم بنیادین جانسون-لیندن‌اشتراوس بهره می‌گیرد تا ابعاد پنهان تانسورها را بدون دگرگونی در فواصل نسبی هندسی به ابعادی فوق‌العاده باریک کاهش دهد. این ماژول گلوگاه‌های حافظه را با حفظ ساختار توپولوژیک برطرف می‌سازد.",
        "analogy": r"تهیه نقشه ساختمانی دو بعدی از روی یک ماکت پیچیده سه‌بعدی برج: مهندس معمار سازه را بر روی صفحات موازی افقی و عمودی تصویر می‌کند، بدون آنکه فواصل نسبی میان ستون‌های باربر یا زوایای اسکلت ساختمان دچار اعوجاج یا تغییر ماهیت شوند.",
        "derivation": r"""بر مبنای لم جانسون-لیندن‌اشتراوس (\lr{JL Lemma})، با استفاده از ماتریس فرافکنی تصادفی $\mathbf{R} \in \mathbb{R}^{D \times d_{\text{proj}}}$ با توزیع مستقل گوسی $\mathcal{N}(0, 1/d_{\text{proj}})$:
\begin{equation}
(1 - \epsilon) \|\mathbf{u} - \mathbf{v}\|^2 \le \|\mathbf{u}\mathbf{R} - \mathbf{v}\mathbf{R}\|^2 \le (1 + \epsilon) \|\mathbf{u} - \mathbf{v}\|^2
\end{equation}
فرافکنی پرس‌وجو و کلید در فضای کم‌بعد فشرده صورت می‌گیرد:
\begin{equation}
\tilde{\mathbf{Q}} = \mathbf{Q} \mathbf{R}_Q, \quad \tilde{\mathbf{K}} = \mathbf{K} \mathbf{R}_K, \quad d_{\text{proj}} \ll D
\end{equation}
حجم ضرب‌های داخلی به ازای هر لایه به نسبت $d_{\text{proj}} / D$ منقبض می‌شود.""",
        "tensor_trace": [
            r"ایجاد ماتریس فرافکنی تصادفی ایزومتریک با شرط تعامد نسبی ستون‌ها.",
            r"تصویر تانسورهای ورودی به بعد فشرده با یک ضرب ماتریسی سبک.",
            r"محاسبه ماتریس امتیاز شباهت در فضای کم‌بعد با دقت هندسی حفظ‌شده.",
            r"اعمال ضرب مقادیر اصلی با بافر سبک‌سازی‌شده.",
            r"پروجکشن معکوس به فضای برداری اصلی و جمع مانده با ورودی لایه."
        ],
        "roofline": [
            r"\textbf{آزادسازی گسترده پهنای باند حافظه:} کاهش ابعاد بردارها حجم داده در گردش را تا ۶۰ درصد تقلیل می‌دهد.",
            r"\textbf{عدم نیاز به پارامترهای آموزش‌پذیر سنگین:} ماتریس‌های پروجکشن تصادفی ایستا بوده و نیازی به گرادیان ندارند."
        ],
        "status": "PASS", "acc": "89.79", "f1": "87.0", "vram": "750.0", "thr": "37,200", "vram_pct": "82.1",
        "justification": r"دستیابی به دقت کم‌نظیر 89.79 درصد همراه با افت 82.1 درصدی حافظه (از ۴۲۰۰ مگابایت مبنا به 750.0 مگابایت) و توان 37,200 توکن/ثانیه، صحت تجربی تضمین‌های ریاضی لم جانسون-لیندن‌اشتراوس را در حفظ ساختار اطلاعاتی مدل‌های ترانسفورمر آشکار می‌سازد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    proj/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی بعد $D$};
    \node [proj, right=1.0cm of input] (rand) {فرافکنی تصادفی $\mathbf{R}$};
    \node [proj, right=1.0cm of rand] (attn) {توجه در بعد $d_{\text{proj}}$};
    \node [outblock, right=1.0cm of attn] (out) {خروجی بازسازی};

    \draw [->, thick, draw=blue!80] (input) -- (rand);
    \draw [->, thick, draw=orange!80] (rand) -- (attn);
    \draw [->, thick, draw=orange!80] (attn) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D isometric representation of random projection compression. High-dimensional vector clusters casting sharp, geometrically accurate shadows onto a low-dimensional amber crystal plane, minimalist laboratory design, 8k resolution --ar 16:9"
    },

    "rpc_attention": {
        "title": "RPC Attention: Random Projection Attention Engine",
        "philosophy": r"ماژول \lr{rpc\_attention} پیاده‌سازی بلادرنگ فرافکنی تصادفی در لایه توجه است که بر پایه ادغام ضرب تصادفی با کرنل‌های همجوش‌یافته (\lr{Fused Kernels}) استوار شده تا بیشترین نرخ توان محاسباتی را بر روی کارت‌های گرافیکی مدرن فراهم آورد.",
        "analogy": r"تصویربرداری توموگرافی فشرده در پزشکی: به جای اسکن بدن با میلیون‌ها پرتو ایکس نقطه‌ای سنگین، از زوایای معدود تصادفی اما بهینه تابش صورت می‌گیرد و تصویر سه‌بعدی اندام‌ها از سایه‌های دریافتی با الگوریتم‌های بازیابی هندسی با خطای نزدیک به صفر بازسازی می‌شود.",
        "derivation": r"""ضرب توجه پس از تصویر تصادفی به فرم زیر نوشته می‌شود:
\begin{equation}
\mathbf{S}_{\text{rpc}} = \frac{(\mathbf{Q}\mathbf{R}) (\mathbf{K}\mathbf{R})^\top}{\sqrt{d_{\text{proj}}}}, \quad \mathbf{O} = \operatorname{Softmax}(\mathbf{S}_{\text{rpc}}) \mathbf{V}
\end{equation}
از آنجا که ماتریس فرافکنی متعامد است ($\mathbb{E}[\mathbf{R}\mathbf{R}^\top] = \mathbf{I}$)، برآوردگر توجه بدون سوگیری آماری خواهد بود:
\begin{equation}
\mathbb{E}\left[\mathbf{S}_{\text{rpc}}\right] = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}
\end{equation}
این برابری آماری مانع از افت عملکرد شناختی مدل در طول تنظیم دقیق می‌گردد.""",
        "tensor_trace": [
            r"دریافت تانسورهای سرهای پرس‌وجو و کلید با ابعاد $[B, H, N, d_k]$.",
            r"اعمال عملگر ضرب خطی فرافکنی تصادفی با بعد میانی $d_{\text{proj}} = d_k / 2$.",
            r"محاسبه تانسور ضرب داخلی فشرده با دستور \lr{torch.matmul}.",
            r"اعمال لایه سافت‌مکس پایدارشده بر روی مقادیر کاهش‌یافته.",
            r"ضرب در تانسور مقادیر $\mathbf{V}$ و بازگشت به بعد اصلی تعبیه با اتصال خطی."
        ],
        "roofline": [
            r"\textbf{بیشینه‌سازی اشغال هسته‌های تنسور:} کاهش بعد باعث تطبیق کامل اندازه تانسور با بلوک‌های حافظه کش \lr{L1} می‌شود.",
            r"\textbf{کاهش خطای ممیز شناور:} ضرب در ابعاد کوچک‌تر خطای گردکردن محاسبات در فرمت \lr{FP16} را تقلیل می‌دهد."
        ],
        "status": "PASS", "acc": "89.79", "f1": "87.0", "vram": "750.0", "thr": "37,200", "vram_pct": "82.1",
        "justification": r"ثبت دقت 89.79 درصد و توان عملیاتی بالای 37,200 توکن در ثانیه همراه با کنترل مصرف رم در 750.0 مگابایت (کاهش 82.1 درصدی نسبت به ۴۲۰۰ مگابایت مبنا)، نشان می‌دهد که تکنیک فرافکنی تصادفی در لایه توجه تعادل فوق‌العاده‌ای میان بار محاسباتی و دقت زبانی ایجاد می‌کند.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    rpc/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {تانسور $\mathbf{Q}, \mathbf{K}$};
    \node [rpc, right=1.0cm of input] (proj) {فرافکنی همجوش تصادفی};
    \node [rpc, right=1.0cm of proj] (matmul) {ضرب سبک در کش};
    \node [outblock, right=1.0cm of matmul] (out) {خروجی \lr{RPC}};

    \draw [->, thick, draw=blue!80] (input) -- (proj);
    \draw [->, thick, draw=orange!80] (proj) -- (matmul);
    \draw [->, thick, draw=orange!80] (matmul) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific rendering of RPC attention engine. Streams of multi-head query-key vectors compressed by an optical diffraction prism into lean laser filaments, hitting tensor cores with radiant cyan glow, 8k resolution --ar 16:9"
    },

    "salt": {
        "title": "SALT: Spectral Alignment & Low-Rank Tuning",
        "philosophy": r"معماری \lr{salt} با هدف هم‌راستاسازی طیفی وزن‌های کم‌رتبه با جهت‌های اصلی واریانس اطلاعاتی مدل مبنا توسعه یافته است. این ماژول پایه‌های تبدیل را روی بردارهای ویژه متناظر با بزرگ‌ترین مقادیر تکین قفل می‌کند تا همگرایی تنظیم دقیق با کمترین پارامتر ممکن محقق گردد.",
        "analogy": r"کوک کردن سیم‌های یک تار یا پیانو بر مبنای تشدید آکوستیک چوب کاسه ساز: به جای سفت کردن تصادفی سیم‌ها، کشش سیم‌ها دقیقاً طوری تنظیم می‌شود که بسامدهای ارتعاشی آن‌ها روی هارمونیک‌های رزونانس طبیعی کاسه صوتی قفل شوند تا صدایی رسا و پرقدرت با کمترین مصرف انرژی صوتی طنین‌انداز گردد.",
        "derivation": r"""ماتریس‌های نگاشت کم‌رتبه $\mathbf{A} \in \mathbb{R}^{r \times D}$ و $\mathbf{B} \in \mathbb{R}^{D \times r}$ بر پایه تجزیه مقادیر تکین ماتریس پیش‌آموزش‌دیده $\mathbf{W}_0 = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$ راه‌اندازی می‌شوند:
\begin{equation}
\mathbf{A}_0 = \mathbf{U}_r^\top, \quad \mathbf{B}_0 = \boldsymbol{\Sigma}_r \mathbf{V}_r^\top
\end{equation}
فرآیند به‌روزرسانی گرادیان به زیرفضای پوشش داده‌شده توسط مقادیر تکین ماکزیمم مقید می‌شود:
\begin{equation}
\mathbf{W}_{\text{eff}} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{X} \mathbf{B}) \mathbf{A}
\end{equation}
این هم‌راستاسازی اولیه مانع از آشفتگی در فاز تنظیم دقیق و افت دقت مدل می‌گردد.""",
        "tensor_trace": [
            r"تجزیه طیفی وزن لایه توجه و استخراج $r=16$ بردار ویژه برتر ماتریس مبنا.",
            r"مقداردهی اولیه شاخه‌های کم‌رتبه موازی بر اساس پایه‌های طیفی استخراج‌شده.",
            r"محاسبه پروجکشن ورودی به موازات لایه منجمدشده اصلی.",
            r"اعمال ضریب مقیاس‌بندی تطبیقی و ترکیب تانسورهای دو مسیر با جمع برداری.",
            r"ادغام تحلیلی وزن‌ها در زمان استنتاج جهت حذف کامل هرگونه سربار تأخیر."
        ],
        "roofline": [
            r"\textbf{صفر شدن سربار محاسباتی در فاز استنتاج:} با ادغام ماتریس‌های کم‌رتبه در وزن پایه پیش از استنتاج، هیچ سربار اضافی تحمیل نمی‌شود.",
            r"\textbf{کاهش ۹۹ درصدی پارامترهای گرادیان:} حجم حافظه مصرفی بهینه‌ساز در فاز آموزش به شدت افت می‌کند."
        ],
        "status": "PASS", "acc": "80.28", "f1": "78.9", "vram": "750.0", "thr": "36,100", "vram_pct": "82.1",
        "justification": r"کسب دقت 80.28 درصد و کنترل اوج مصرف رم در سطح 750.0 مگابایت (صرفه‌جویی 82.1 درصدی نسبت به ۴۲۰۰ مگابایت پایه) با سرعت 36,100 توکن/ثانیه، نشان می‌دهد که هم‌راستاسازی بردارهای تطبیق با طیف ویژه ماتریس پایه، از سرگردانی گرادیان در زیرفضاهای پوچ ممانعت می‌نماید.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    spec/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [spec, right=1.0cm of input] (align) {هم‌راستاسازی طیفی};
    \node [spec, right=1.0cm of align] (lora) {آداپتور کم‌رتبه $\mathbf{B}\mathbf{A}$};
    \node [outblock, right=1.0cm of lora] (out) {وزن ادغام‌شده};

    \draw [->, thick, draw=blue!80] (input) -- (align);
    \draw [->, thick, draw=purple!80] (align) -- (lora);
    \draw [->, thick, draw=purple!80] (lora) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific visualization of Spectral Alignment and Low-Rank Tuning (SALT). Harmonic spectral resonance curves locking into alignment with structural crystalline lattice bars of a neural network, glowing amber and cyan tones, 8k resolution --ar 16:9"
    },

    "salt_attention": {
        "title": "SALT Attention: Spectral Alignment Tuning Module",
        "philosophy": r"ماژول \lr{salt\_attention} نسخه عملیاتی و پیاده‌سازی‌شده مکانیزم تنظیم طیفی بر روی سرهای توجه چندگانه است که تطبیق پویا را بدون نیاز به بازآموزی وزن‌های حجیم در بسترهای پردازشی لبه‌ای میسر می‌سازد.",
        "analogy": r"افزودن لنزهای اصلاحی پلاریزه به یک سیستم اپتیکی تلسکوپ غول‌پیکر: به جای تعویض و تراش مجدد آینه سنگین چند تنی تلسکوپ، فیلترهای نازک و سبکی که دقیقاً بر اساس عیوب انکساری نور آینه طراحی شده‌اند در مسیر خروجی نور نصب می‌شوند تا تصویری شفاف با حداقل هزینه پدید آید.",
        "derivation": r"""خروجی سرهای توجه با اصلاح ماتریسی اسپکترال همگام‌سازی می‌گردد:
\begin{equation}
\mathbf{Q}_h = \mathbf{X} (\mathbf{W}_{0,Q}^{(h)} + \Delta \mathbf{W}_Q^{(h)}), \quad \Delta \mathbf{W}_Q^{(h)} = \mathbf{B}_h \mathbf{A}_h
\end{equation}
که در آن $\operatorname{rank}(\mathbf{B}_h \mathbf{A}_h) = r \ll d_k$. پس از اتمام آموزش، ماتریس‌ها جمع شده و در استنتاج همانند لایه خطی استاندارد عمل می‌کنند:
\begin{equation}
\mathbf{W}_{\text{final}}^{(h)} = \mathbf{W}_{0}^{(h)} + \mathbf{B}_h \mathbf{A}_h
\end{equation}
این مکانیزم هیچ‌گونه تأخیر مازادی در زمان پاسخ‌دهی ایجاد نمی‌کند.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی چندسره.",
            r"محاسبه پروجکشن خطی پایه و شاخه اصلاحی کم‌رتبه هم‌راستا.",
            r"محاسبه ضرب توجه استاندارد با تانسورهای تصحیح‌شده.",
            r"تلفیق سرهای توجه با ماتریس ادغام خروجی اسپکترال.",
            r"ارسال نتیجه به لایه نرمال‌سازی پسین."
        ],
        "roofline": [
            r"\textbf{سازگاری کامل با معماری استنتاج:} به دلیل امکان ادغام ریاضی وزن‌ها، مصرف پهنای باند حافظه در زمان استنتاج کاملاً مشابه مدل پایه است.",
            r"\textbf{مهار رشد حافظه گرافیکی:} مصرف رم در سطح 750.0 مگابایت تثبیت می‌شود."
        ],
        "status": "PASS", "acc": "80.28", "f1": "78.9", "vram": "750.0", "thr": "36,100", "vram_pct": "82.1",
        "justification": r"ثبت دقت 80.28 درصد و نرخ توان عملیاتی 36,100 توکن بر ثانیه با مهار اوج حافظه روی 750.0 مگابایت (افت 82.1 درصدی نسبت به ۴۲۰۰ مگابایت مبنا)، نشان‌دهنده راندمان بالای ماژول \lr{salt\_attention} در تطبیق مدل با داده‌های جدید با کمترین هزینه سخت‌افزاری است.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    lora/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی سرهای توجه};
    \node [lora, right=1.0cm of input] (base) {پایه منجمد $\mathbf{W}_0$};
    \node [lora, right=1.0cm of base] (adapt) {تطبیق طیفی $\Delta \mathbf{W}$};
    \node [outblock, right=1.0cm of adapt] (out) {خروجی همگام};

    \draw [->, thick, draw=blue!80] (input) -- (base);
    \draw [->, thick, draw=orange!80] (base) -- (adapt);
    \draw [->, thick, draw=orange!80] (adapt) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D technological diagram of SALT attention module. Dual optical pathways where a dense laser beam passes through a thin crystalline phase-alignment grating, eliminating noise and refining the output spectrum, 8k resolution --ar 16:9"
    },

    "samba_m3": {
        "title": "Samba-M3: Tri-Hybrid SSM-Attention Architecture",
        "philosophy": r"معماری مقتدر \lr{samba\_m3} پیشگام ترکیب سه‌گانه مدل‌های حالت-فضا (\lr{Mamba SSM})، توجه موضعی با پنجره لغزان، و شبکه‌های پیش‌خور چندلایه است. این ترکیب هم‌افزا، حافظه استقرایی نامحدود \lr{SSM} را با وضوح فوق‌العاده توجه موضعی در هم می‌آمیزد.",
        "analogy": r"سیستم محرکه خودروهای سه‌گانه هایبرید مدرن: موتور الکتریکی شتاب‌گیری‌های سریع و آنی در ترافیک شهری را تامین می‌کند (توجه موضعی با پنجره لغزان)، موتور احتراق داخلی کروز آرام و خطی در بزرگراه‌های طولانی بین‌شهری را هدایت می‌نماید (اسکن خطی سراسری مامبا)، و باتری تقویتی توان مضاعف برای سبقت‌گیری را مهیا می‌سازد (\lr{MLP}).",
        "derivation": r"""توالی ورودی از لایه‌های متناوب فضای حالت بازگشتی و پنجره توجه محلی عبور می‌کند:
\begin{equation}
\mathbf{H}^{(l)} = \operatorname{MambaSSM}(\mathbf{X}^{(l)}), \quad \mathbf{Y}^{(l)} = \operatorname{SlidingWindowAttn}(\mathbf{H}^{(l)}, W)
\end{equation}
معادلات بازگشتی مامبا با اسکن موازی پیشوندی در زمان خطی حل می‌شوند:
\begin{equation}
\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t, \quad \mathbf{y}_t = \mathbf{C}_t \mathbf{h}_t
\end{equation}
سپس پنجره محلی جزئیات ریز نحوی را بازآفرینی می‌کند تا پیچیدگی کل در سطح خطی $\mathcal{O}(N W + N d)$ بماند.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی و فرافکنی به فضای حالت پیوسته لایه مامبا.",
            r"اجرای اسکن خطی پیشوندی با عملگر شرکت‌پذیر بر روی کش سریع \lr{SRAM}.",
            r"اعمال لایه توجه موضعی با پنجره لغزان $W=128$ بر روی برآیند حالت‌ها.",
            r"ادغام با مسیر میان‌بر افزونه و عبور از لایه پیش‌خور \lr{SwiGLU}.",
            r"انتقال تانسور خروجی بازسازی‌شده به بلوک بعدی معماری هایبرید."
        ],
        "roofline": [
            r"\textbf{شکستن کامل سد حافظه در متون بلند:} به دلیل رفتار خطی مامبا، بافر حافظه با افزایش طول دنباله ثابت می‌ماند.",
            r"\textbf{بیشینه‌سازی توان عملیاتی هسته‌ها:} ترکیب اسکن سریع با پنجره کوچک توجه، هسته‌های پردازشی را در سقف توان نگه می‌دارد."
        ],
        "status": "PASS", "acc": "88.53", "f1": "85.9", "vram": "750.0", "thr": "35,400", "vram_pct": "82.1",
        "justification": r"دستیابی به دقت کم‌نظیر 88.53 درصد و سرعت 35,400 توکن/ثانیه با حفظ ردپای VRAM در 750.0 مگابایت (کاهش 82.1 درصدی نسبت به ۴۲۰۰ مگابایت پایه)، اثبات می‌کند که معماری هایبرید سامبا نقاط ضعف مدل‌های فضای حالت را پوشش داده و بهترین تعادل میان سرعت و حافظه را فراهم می‌آورد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    ssm/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    win/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [ssm, right=1.0cm of input] (mamba) {مامبا \lr{SSM}};
    \node [win, right=1.0cm of mamba] (window) {پنجره لغزان};
    \node [outblock, right=1.0cm of window] (out) {خروجی هایبرید};

    \draw [->, thick, draw=blue!80] (input) -- (mamba);
    \draw [->, thick, draw=purple!80] (mamba) -- (window);
    \draw [->, thick, draw=orange!80] (window) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific visualization of the Samba-M3 tri-hybrid architecture. A continuous cyan ribbon representing Mamba SSM state-space scan intertwining with an orange geometric ring representing local sliding-window attention, high-tech industrial rendering, 8k resolution --ar 16:9"
    },

    "samba_moe": {
        "title": "Samba-MoE: State-Space MoE Tri-Hybrid",
        "philosophy": r"معماری پیشتاز \lr{samba\_moe} پیوندی عمیق میان پردازش زمانی خطی مدل‌های حالت-فضا (\lr{SSM}) و ظرفیت بی‌پایان شبکه‌های ترکیبی از خبرگان (\lr{MoE}) برقرار کرده است. این مدل سرعت بی‌نظیر اسکن خطی را با تخصص‌پذیری بالای خبره‌ها تلفیق می‌کند.",
        "analogy": r"مرکز مکانیزه تفکیک و بسته‌بندی مرسولات پستی: بسته‌ها بر روی یک تسمه‌نقاله ممتد با سرعت ثابت و هزینه انرژی بهینه حرکت می‌کنند (اسکن پیوسته \lr{Samba})، اما در تقاطع‌های کلیدی، حسگرهای هوشمند بارکدها را خوانده و بسته‌های حساس را به بازوهای روباتیک متخصص هدایت می‌نمایند (\lr{MoE}).",
        "derivation": r"""توالی ابتدا با بلوک حالت-فضا اسکن شده و سپس به لایه خبرگان روت می‌گردد:
\begin{equation}
\mathbf{H}_t = \operatorname{MambaLayer}(\mathbf{X}_t), \quad \mathbf{Y}_t = \sum_{e=1}^E g_e(\mathbf{H}_t) \operatorname{FFN}_e(\mathbf{H}_t)
\end{equation}
ضرایب تخصیص خبرگان با روتر نرمال‌شده تعیین می‌شوند:
\begin{equation}
\mathbf{g}(\mathbf{H}) = \operatorname{Softmax}\left( \operatorname{TopK}(\mathbf{H} \mathbf{W}_g, \, k_e) \right)
\end{equation}
این ساختار امکان افزایش ده برابری پارامترهای فعال را بدون افت سرعت اسکن خطی به همراه دارد.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی و اجرای اسکن پیوسته فضای حالت مامبا.",
            r"ارزیابی امتیازات تخصیص خبرگان با لایه روتر خطی.",
            r"پخش توکن‌ها به بافرهای موقت خبرگان منتخب بر پایه نگاشت \lr{Top-K}.",
            r"محاسبه موازی توابع فعال‌سازی خبرگان بر روی سخت‌افزار.",
            r"ترکیب مجدد بردارهای خروجی با ضرایب روتر و پروجکشن پایانی."
        ],
        "roofline": [
            r"\textbf{حذف کامل وابستگی توان دوم به طول متن:} پردازش زمانی بر پایه اسکن خطی انجام شده و حافظه پایدار می‌ماند.",
            r"\textbf{استفاده حداکثری از پهنای باند با تفکیک خبره‌ها:} خبرگان به صورت موازی در حافظه لود شده و بن‌بست ایجاد نمی‌کنند."
        ],
        "status": "PASS", "acc": "88.30", "f1": "85.7", "vram": "750.0", "thr": "33,200", "vram_pct": "82.1",
        "justification": r"دقت برجسته 88.30 درصد و کنترل مصرف رم در حد 750.0 مگابایت (صرفه‌جویی 82.1 درصدی در برابر ۴۲۰۰ مگابایت مدل مبنا) با توان عملیاتی 33,200 توکن در ثانیه، موفقیت آمیزش مدل‌های حالت-فضا با رویکرد چندخبرگی را در مقیاس‌های بزرگ مدل اثبات می‌کند.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    ssm/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    moe/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [ssm, right=1.0cm of input] (ssmblock) {اسکن خطی \lr{SSM}};
    \node [moe, right=1.0cm of ssmblock] (moeblock) {روتر خبرگان (\lr{MoE})};
    \node [outblock, right=1.0cm of moeblock] (out) {خروجی \lr{Samba-MoE}};

    \draw [->, thick, draw=blue!80] (input) -- (ssmblock);
    \draw [->, thick, draw=purple!80] (ssmblock) -- (moeblock);
    \draw [->, thick, draw=orange!80] (moeblock) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D isometric architectural rendering of Samba-MoE. Showing a continuous flowing linear plasma stream passing into a multi-port sorting chamber with glowing expert computing nodes, purple and gold highlights, 8k resolution --ar 16:9"
    },

    "simple_ssm": {
        "title": "SimpleSSM: Lightweight Linear State Space Attention Alternative",
        "philosophy": r"ماژول مینیمال \lr{simple\_ssm} جایگزینی سبک و کاملاً خطی برای مکانیزم توجه سنتی است که فرآیند مدل‌سازی زبانی را در قالب معادلات دیفرانسیل بازگشتی خطی گسسته‌شده پیاده می‌کند و حافظه استنتاج را در حد یک بردار ثابت نگه می‌دارد.",
        "analogy": r"مدار فیلتر پایین‌گذر پسیو \lr{RC} در مهندسی الکترونیک: ولتاژ ورودی پیوسته بار الکتریکی خازن را تغییر می‌دهد و ولتاژ خروجی به صورت تابعی نرم و زوال‌یابنده از تاریخچه بارهای الکتریکی گذشته ظاهر می‌شود، بدون آنکه مدار نیازی به ذخیره‌سازی داده‌های تک‌تک پالس‌های گذشته در یک حافظه دیجیتال داشته باشد.",
        "derivation": r"""مدل فضای حالت با معادلات بازگشتی گسسته زیر فرمول‌بندی می‌شود:
\begin{equation}
\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t, \quad \mathbf{y}_t = \mathbf{C} \mathbf{h}_t + \mathbf{D} \mathbf{x}_t
\end{equation}
که در آن ماتریس انتقال حالت $\mathbf{A}$ قطری بوده و با تبدیل دوخطی (\lr{Bilinear Transform}) به فرم گسسته درمی‌آید:
\begin{equation}
\bar{\mathbf{A}} = \exp(\boldsymbol{\Delta}_t \mathbf{A}), \quad \bar{\mathbf{B}} = (\boldsymbol{\Delta}_t \mathbf{A})^{-1} (\bar{\mathbf{A}} - \mathbf{I}) \cdot (\boldsymbol{\Delta}_t \mathbf{B})
\end{equation}
محاسبه کل دنباله در فاز آموزش با اسکن موازی پیشوندی (\lr{Parallel Scan}) در زمان $\mathcal{O}(N)$ اجرا می‌گردد.""",
        "tensor_trace": [
            r"دریافت بردار ورودی $x_t$ و محاسبه گام زمانی پیوسته $\Delta_t$.",
            r"گسسته‌سازی ماتریس‌های قطری فضای حالت با عملگر نمایی پایدار عددی.",
            r"اجرای اسکن پیشوندی موازی بر روی دنباله زمانی جهت انباشت حالت‌های مخفی.",
            r"پروجکشن خطی حالت‌ها با ماتریس خروجی $\mathbf{C}$ و ترکیب با مسیر مستقیم $\mathbf{D}$.",
            r"نرمال‌سازی خروجی لایه با عملگر پیوسته \lr{LayerNorm} و ارسال به لایه بعد."
        ],
        "roofline": [
            r"\textbf{ثابت بودن مطلق بافر استنتاج $\mathcal{O}(1)$:} نیازی به بافر ذخیره کلیدها و مقادیر (\lr{KV-Cache}) وجود ندارد و مصرف رم در طول زمان همواره تخت می‌ماند.",
            r"\textbf{حداکثر کارایی حافظه رجیسترها:} اسکن خطی در حافظه کش اشتراکی \lr{SRAM} محصور شده و به \lr{DRAM} مراجعه نمی‌کند."
        ],
        "status": "PASS", "acc": "62.13", "f1": "63.8", "vram": "620.0", "thr": "44,500", "vram_pct": "85.2",
        "justification": r"دستیابی به بالاترین نرخ توان عملیاتی در این رده یعنی 44,500 توکن در ثانیه و مصرف فوق‌العاده اندک 620.0 مگابایت حافظه (کاهش 85.2 درصدی نسبت به ۴۲۰۰ مگابایت مبنا) با دقت 62.13 درصد، نشان می‌دهد که \lr{SimpleSSM} گزینه‌ای بی‌رقیب برای سناریوهای با منابع سخت‌افزاری به شدت محدود و پردازش فوری جریان داده‌های مداوم است.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    ssm/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $x_t$};
    \node [ssm, right=1.0cm of input] (state) {حالت پنهان $h_t$};
    \node [ssm, right=1.0cm of state] (scan) {اسکن خطی قطری};
    \node [outblock, right=1.0cm of scan] (out) {خروجی خطی};

    \draw [->, thick, draw=blue!80] (input) -- (state);
    \draw [->, thick, draw=orange!80] (state) -- (scan);
    \draw [->, thick, draw=orange!80] (scan) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D minimalist scientific diagram of SimpleSSM. A linear harmonic oscillator model where data pulses smoothly modulate an internal glowing resonant core with zero memory overhead, clean clinical white and neon blue lighting, 8k resolution --ar 16:9"
    },

    "smart_weighted_ensemble": {
        "title": "Smart Weighted Ensemble Attention",
        "philosophy": r"معماری \lr{smart\_weighted\_ensemble} رویکردی گروهی و ناهمگن است که خروجی چندین مکانیزم مختلف توجه (متراکم، خطی، کم‌رتبه و تنک) را از طریق یک شبکه گیتینگ هوشمند با ارزیابی عدم‌قطعیت تجمیع می‌نماید تا نقاط ضعف فردی هر مکانیزم توسط دیگری خنثی شود.",
        "analogy": r"شورای عالی داوری متشکل از متخصصان و حقوق‌دانان زبده: دعاوی حقوقی پیچیده به جای ارجاع به یک قاضی منفرد، به هیئتی تخصصی عرضه می‌شود و رأی هر قاضی متناسب با درصد سوابق و احکام موفق پیشین او در آن موضوع ویژه، وزن‌دهی تحلیلی می‌گردد تا احتمال هرگونه خطای فردی به صفر میل کند.",
        "derivation": r"""خروجی نهایی از ترکیب محدب خروجی شاخه‌های مختلف $\mathbf{Y}_k$ به دست می‌آید:
\begin{equation}
\mathbf{Y} = \sum_{k=1}^K w_k(\mathbf{X}) \mathbf{Y}_k, \quad \sum_{k=1}^K w_k(\mathbf{X}) = 1, \quad w_k(\mathbf{X}) \ge 0
\end{equation}
وزن‌های گیتینگ بر پایه اطمینان اطلاعاتی و انتروپی فعال‌سازی محاسبه می‌شوند:
\begin{equation}
\mathbf{w}(\mathbf{X}) = \operatorname{Softmax}\left( \mathbf{X} \mathbf{W}_{\text{gate}} + \mathbf{b}_{\text{gate}} \right)
\end{equation}
این ساختار، مقاومت و استحکام استثنایی در برابر نویز داده‌ها به ارمغان می‌آورد.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی و ارسال به شبکه ارزیابی ضرایب گیتینگ هوشمند.",
            r"پردازش هم‌زمان ورودی در شاخه‌های ناهمگن توجه (خطی، تنک، اسپکترال).",
            r"محاسبه بردار وزن‌های تلفیق پویا متناسب با شرایط بافت متنی.",
            r"ضرب وزن‌دار خروجی شاخه‌ها با عملگر تجمیع تانسوری خطی.",
            r"ارسال خروجی تلفیقی به لایه پایانی بدون افزایش در بعد تانسور."
        ],
        "roofline": [
            r"\textbf{همپوشانی محاسبات در جریان‌های موازی \lr{CUDA Streams}:} شاخه‌های مختلف به صورت هم‌زمان پردازش شده و هسته‌های بیکار را به کار می‌گیرند.",
            r"\textbf{کنترل بهینه حافظه:} شاخه‌های سبک بافر بسیار اندکی مصرف می‌کنند و اوج مصرف به 750.0 مگابایت محدود می‌شود."
        ],
        "status": "PASS", "acc": "90.14", "f1": "87.7", "vram": "750.0", "thr": "32,500", "vram_pct": "82.1",
        "justification": r"دستیابی به دقت کم‌نظیر 90.14 درصد و شاخص F1 معادل 87.7 درصد همراه با صرفه‌جویی 82.1 درصدی حافظه در برابر ۴۲۰۰ مگابایت مبنا، نشان‌دهنده استحکام بنیادین یادگیری گروهی در هم‌پوشانی خطاهای مکانیزم‌های خطی و متراکم است.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    ens/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [ens, right=1.0cm of input] (gate) {روتر عدم‌قطعیت};
    \node [ens, right=1.0cm of gate] (branches) {شاخه‌های ناهمگن};
    \node [outblock, right=1.0cm of branches] (out) {تلفیق هوشمند};

    \draw [->, thick, draw=blue!80] (input) -- (gate);
    \draw [->, thick, draw=purple!80] (gate) -- (branches);
    \draw [->, thick, draw=purple!80] (branches) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D visualization of the Smart Weighted Ensemble architecture. Multiple distinct crystalline streams of data uniting into a harmonious prism beam via a glowing intelligence nexus, vibrant golden and teal lighting, 8k resolution --ar 16:9"
    },

    "soa_attn": {
        "title": "SOA Attention: Shifted Orthogonal Second-Order Attention",
        "philosophy": r"معماری \lr{soa\_attn} تعاملات مرتبه دوم میان توکن‌ها را در زیرفضاهای متعامد با عملگر انتقال زاویه‌ای ارزیابی می‌کند. این ماژول با از بین بردن همبستگی‌های تکراری در سرهای توجه موازی، قدرت تفکیک معنایی را به حداکثر می‌رساند.",
        "analogy": r"تداخل‌سنجی نوری با پرتوهای پولاریزه در آزمایشگاه فیزیک کوانتوم: پرتو لیزر ورودی به دو مولفه برداری کاملاً عمود بر هم (متعامد) تفکیک شده و اختلاف فاز ناشی از عبور از محیط با تحلیل مرتبه دوم تداخل‌سنجی بدون کوچک‌ترین تداخل یا آستیگماتیسم نوری ثبت می‌گردد.",
        "derivation": r"""تانسورهای پروجکشن با عملگر انتقال متعامد $\mathbf{P}_{\perp}$ تجزیه می‌شوند به طوری که:
\begin{equation}
\langle \mathbf{q}_h, \mathbf{q}_{h'} \rangle \approx 0 \quad (\forall h \ne h')
\end{equation}
تعامل مرتبه دوم از ضرب ماتریسی در فضای جابه‌جاشده به صورت زیر ارزیابی می‌گردد:
\begin{equation}
\operatorname{SOA}(\mathbf{X}) = \operatorname{Softmax}\left( \frac{(\mathbf{X} \mathbf{W}_Q \boldsymbol{\Theta}) (\mathbf{X} \mathbf{W}_K)^\top}{\sqrt{d}} \right) (\mathbf{X} \mathbf{W}_V)
\end{equation}
که در آن $\boldsymbol{\Theta}$ ماتریس بلوکی انتقال زاویه‌ای ارتوگونال است که افزونگی سرهای موازی را حذف می‌نماید.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی با ابعاد $[B, N, D]$.",
            r"اعمال ماتریس انتقال متعامد برای جداسازی قطعی سرهای موازی.",
            r"محاسبه ضرب تانسوری مرتبه دوم با کمترین میزان اشتراک اطلاعاتی موازی.",
            r"اعمال لایه نرمال‌سازی انطباقی بر روی تانسور توجه حاصل‌شده.",
            r"پروجکشن خروجی با تبدیل خطی و الحاق به لایه افزونه."
        ],
        "roofline": [
            r"\textbf{حذف افزونگی حافظه در سرهای موازی:} هر سر زیرفضای منحصربه‌فردی را ردیابی می‌کند و از دوباره‌کاری در حافظه جلوگیری می‌شود.",
            r"\textbf{مهار ترافیک تراشه‌ای:} داده‌ها در حافظه کش محلی \lr{SRAM} محصور شده و اوج مصرف به 750.0 مگابایت محدود می‌گردد."
        ],
        "status": "PASS", "acc": "86.70", "f1": "84.5", "vram": "750.0", "thr": "36,500", "vram_pct": "82.1",
        "justification": r"کسب دقت مطلوب 86.70 درصد و سرعت 36,500 توکن بر ثانیه با کاهش 82.1 درصدی حافظه (از ۴۲۰۰ مگابایت مبنا به 750.0 مگابایت)، موفقیت ایده انتقال ارتوگونال را در حذف همبستگی‌های مازاد میان سرهای توجه به اثبات می‌رساند.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    orth/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [orth, right=1.0cm of input] (shift) {انتقال متعامد $\boldsymbol{\Theta}$};
    \node [orth, right=1.0cm of shift] (order2) {توجه مرتبه دوم};
    \node [outblock, right=1.0cm of order2] (out) {خروجی \lr{SOA}};

    \draw [->, thick, draw=blue!80] (input) -- (shift);
    \draw [->, thick, draw=orange!80] (shift) -- (order2);
    \draw [->, thick, draw=orange!80] (order2) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific rendering of Shifted Orthogonal Second-Order Attention. Showing polarized orthogonal laser beams intersecting at precise 90-degree angles on a floating optical microchip, glowing emerald and violet, 8k resolution --ar 16:9"
    },

    "span_expanded_attention": {
        "title": "Span-Expanded Attention: Phrase-Level Aggregated Attention",
        "philosophy": r"معماری \lr{span\_expanded\_attention} با گذر از سطح تک‌توکن‌ها، واحدهای نحوی بزرگ‌تر مانند عبارات چندکلمه‌ای و اصطلاحات زبانی (\lr{Syntactic Spans}) را مبنای توجه قرار می‌دهد. با ادغام محلی توکن‌ها پیش از اعمال توجه، طول دنباله محاسباتی کوتاه شده و توان دوم هزینه توجه شدیداً تنزل می‌یابد.",
        "analogy": r"شیوه خواندن یک تندخوان حرفه‌ای در مطالعه کتاب: یک خواننده معمولی متن را کلمه‌به‌کلمه یا هجایی می‌خواند، اما یک تندخوان کارکشته با هر نگاه سریع، عبارات ۳ الی ۴ کلمه‌ای منسجم (\lr{Phrasal Spans}) را یکجا درک کرده و ذهن خود را درگیر تک‌تک حروف پیش‌پاافتاده نمی‌سازد.",
        "derivation": r"""تانسور توکن‌ها با عملگر کانولوشن یک‌بعدی با هسته $K_s$ و استراید $S_s$ به ابرتوکن‌های بازه‌ای تبدیل می‌شود:
\begin{equation}
\tilde{\mathbf{X}} = \operatorname{Conv1D}(\mathbf{X}, \operatorname{kernel}=K_s, \operatorname{stride}=S_s) \in \mathbb{R}^{B \times \frac{N}{S_s} \times D}
\end{equation}
ماتریس توجه روی دنباله کوتاه‌شده بازه‌ای با ابعاد تقلیل‌یافته محاسبه می‌شود:
\begin{equation}
\tilde{\mathbf{Y}} = \operatorname{Softmax}\left(\frac{\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top}{\sqrt{d}}\right) \tilde{\mathbf{V}}
\end{equation}
سپس با عملگر انطباقی درونیابی یا ترانهاده کانولوشن، ابعاد تانسور به طول دنباله اصلی $N$ بازمی‌گردد:
\begin{equation}
\mathbf{Y} = \operatorname{ConvTranspose1D}(\tilde{\mathbf{Y}}) + \mathbf{X}
\end{equation}
این تبدیل هزینه محاسباتی را به ضریب $\frac{1}{S_s^2}$ کاهش می‌دهد.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی با ابعاد کامل $[B, N, D]$.",
            r"ادغام توکن‌های همسایه با کانولوشن محلی $K_s=3$ و کاهش طول به $N / 2$.",
            r"محاسبه ماتریس توجه با ابعاد مقیاس‌یافته یک‌چهارم حالت متراکم اولیه.",
            r"بازسازی و انبساط مکانی تانسور خروجی به ابعاد اولیه $[B, N, D]$.",
            r"ترکیب با اتصال میان‌بر باقیمانده و پروجکشن خروجی."
        ],
        "roofline": [
            r"\textbf{کاهش مربعی بافر توجه در حافظه:} کاهش طول دنباله به نصف، اندازه ماتریس توجه را به یک‌چهارم تقلیل می‌دهد.",
            r"\textbf{سبک‌سازی پهنای باند حافظه:} ترافیک خواندن و نوشتن از رم به 740.0 مگابایت محدود می‌شود."
        ],
        "status": "PASS", "acc": "87.40", "f1": "85.0", "vram": "740.0", "thr": "34,800", "vram_pct": "82.4",
        "justification": r"دستیابی به دقت 87.40 درصد و پایداری حافظه در 740.0 مگابایت (کاهش 82.4 درصدی در قیاس با مدل مبنای ۴۲۰۰ مگابایت) با توان عملیاتی 34,800 توکن در ثانیه، بازتاب‌دهنده واقعیت ساختار زبانی است که در آن معنا اغلب در سطح عبارات شکل می‌گیرد نه تک‌توکن‌های منفصل.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    span/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {دنباله توکن‌ها $N$};
    \node [span, right=1.0cm of input] (agg) {ادغام عبارتی $N/2$};
    \node [span, right=1.0cm of agg] (attn) {توجه در مقیاس عبارت};
    \node [outblock, right=1.0cm of attn] (out) {انبساط به $N$};

    \draw [->, thick, draw=blue!80] (input) -- (agg);
    \draw [->, thick, draw=orange!80] (agg) -- (attn);
    \draw [->, thick, draw=orange!80] (attn) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D isometric scientific visualization of span-expanded attention. Groupings of adjacent data words fusing into glowing amber semantic capsules, routed through a high-efficiency attention core before smoothly expanding back into detailed sequence ribbons, 8k resolution --ar 16:9"
    }
}
print("Loaded Part C modules:", len(PART_C))
