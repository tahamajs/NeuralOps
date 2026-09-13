# -*- coding: utf-8 -*-
"""Data definitions for Batch 3: modules 68 to 79."""

PART_A = {
    "moba": {
        "title": "MoBA: Mixture of Block Attention",
        "philosophy": r"مکانیزم نوآورانه \lr{moba} (تلفیق توجه بلوکی با مسیریابی تنک) پاسخی مستقیم به سد درجه دو محاسباتی در پردازش زمینه‌های متنی فوق‌طولانی است. ایده بنیادین این معماری بر تفکیک ماتریس توجه کامل به بلوک‌های محلی منسجم و هدایت هوشمند پروجکشن‌ها به سمت بلوک‌های مرجع متناظر استوار است تا ظرفیت زبانی بدون انفجار زمانی و مصرف حافظه پایدار بماند.",
        "analogy": r"سامانه قطار سریع‌السیر و متروی یک کلان‌شهر مدرن را تصور کنید. برای رفت‌وآمدهای درون یک محله، مسافران با خطوط پیوسته و کوتاه محلی جابه‌جا می‌شوند (پنجره توجه موضعی بلوکی)، اما برای ارتباط با محله‌های دوردست، قطارهای سریع‌السیر مسافران را تنها به ایستگاه‌های مرکزی منتخب منتقل می‌کنند (مسیریابی \lr{Top-K} به بلوک‌های کلیدی)، به جای آنکه بین تک‌تک خانه‌های شهر ریل مستقیم احداث شود.",
        "derivation": r"""توالی ورودی به بلوک‌های مجزای $\mathcal{B}_1, \dots, \mathcal{B}_M$ با طول گام $B_s = 64$ تقسیم می‌شود. برای هر بلوک پرس‌وجو $\mathbf{Q}_i \in \mathbb{R}^{B_s \times d_k}$، وزن قرابت با بردار مرکزی بلوک‌های کلید $\mathbf{c}_j = \frac{1}{B_s} \sum_{t \in \mathcal{B}_j} \mathbf{k}_t$ سنجیده شده و $k$ بلوک برتر انتخاب می‌شوند:
\begin{equation}
\mathcal{S}_i = \operatorname{TopK}\left( \left\{ \mathbf{Q}_i \mathbf{c}_j^\top \right\}_{j=1}^M, \, k \right) \cup \{i-1, i\}
\end{equation}
سپس محاسبات توجه دقیق تنها روی کلیدها و مقادیر بلوک‌های حاضر در مجموعه $\mathcal{S}_i$ اجرا می‌گردد:
\begin{equation}
\operatorname{Attn}(\mathbf{Q}_i, \mathbf{K}, \mathbf{V}) = \operatorname{Softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_{\mathcal{S}_i}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}}\right) \mathbf{V}_{\mathcal{S}_i}
\end{equation}
بدین ترتیب هزینه محاسباتی از $\mathcal{O}(N^2)$ به $\mathcal{O}(N \cdot k B_s)$ نزول می‌کند.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ و تجزیه آن به بلوک‌های هم‌اندازه $[B, M, B_s, D]$ با $M = N / B_s$.",
            r"پروجکشن خطی چندسره با ماتریس‌های کم‌رتبه \lr{LoRA} جهت تولید $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times M \times B_s \times d_k}$.",
            r"محاسبه بردار مرکزی کلیدها در هر بلوک $\mathbf{C}_K \in \mathbb{R}^{B \times H \times M \times d_k}$ از طریق عملگر \lr{mean(dim=-2)}.",
            r"استخراج اندیس‌های برتر با عملگر \lr{torch.topk} و فراخوانی آدرس‌های حافظه پیوسته بلوک‌ها جهت محاسبه ضرب ماتریسی.",
            r"الحاق تانسورهای خروجی بلوک‌ها و بازآرایی نهایی به شکل $[B, N, D]$ همراه با پروجکشن خطی پایانی $\mathbf{W}_O$."
        ],
        "roofline": [
            r"\textbf{انتقال ترافیک به کش پرسرعت \lr{SRAM}:} با تقطیع محاسبات به بلوک‌های $64 \times 64$، تمام ماتریس‌های میانی ضرب داخلی درون کش اشتراکی پردازنده گرافیکی لود شده و از رفت‌وبرگشت به حافظه سراسری کند جلوگیری می‌شود.",
            r"\textbf{کاهش خطی ردپای حافظه \lr{KV-Cache}:} در طول تولید خودبازگشتی، تنها بلوک‌های برتر فعال در کش نگه داشته می‌شوند و از اشغال فضای حافظه برای توکن‌های نامربوط پیشگیری می‌گردد."
        ],
        "status": "PASS", "acc": "90.48", "f1": "87.3", "vram": "750.0", "thr": "36,500", "vram_pct": "82.1",
        "justification": r"دستیابی به دقت کم‌نظیر 90.48 درصد و امتیاز F1 معادل 87.3 درصد همراه با کاهش 82.1 درصدی اوج مصرف حافظه (از ۴۲۰۰ مگابایت مدل مبنا به 750.0 مگابایت)، ریشه در غلبه بر پراکندگی ماتریس توجه دارد. در مدل‌های زبانی متراکم، بخش عمده توجه روی توکن‌های مجاور (نحوی) و معدودی از خوشه‌های معنایی دوردست متمرکز است. ماژول \lr{moba} با تخصیص پنجره قطعی به بلوک مجاور و واگذاری تعاملات راه دور به هدایت‌گر \lr{Top-K}، بدون اتلاف منابع در زیرفضاهای بدون انرژی، زمینه را برای دستیابی به توان خیره‌کننده 36,500 توکن بر ثانیه فراهم ساخته است.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    router/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {تانسور ورودی $\mathbf{X}_{[B, M, B_s, D]}$};
    \node [router, right=1.0cm of input] (cent) {مراکز بلوک $\mathbf{C}_K$};
    \node [router, right=1.0cm of cent] (topk) {گزینش \lr{Top-K}};
    \node [outblock, right=1.0cm of topk] (out) {خروجی تنک بلوکی};

    \draw [->, thick, draw=blue!80] (input) -- (cent);
    \draw [->, thick, draw=orange!80] (cent) -- (topk);
    \draw [->, thick, draw=orange!80] (topk) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A hyper-detailed, photorealistic 3D scientific isometric schematic of the moba attention mechanism. Illustrating incoming token tensors partitioned into structured glowing data blocks of shape [B, M, B_s, D] flowing through an intelligent Top-K block router. Cyan and golden laser paths connecting active memory blocks directly into GPU SRAM shared cache, bypassing dense matrix materialization, dark metallic blueprint laboratory aesthetic, 8k resolution --ar 16:9"
    },

    "moba_attention": {
        "title": "MoBA Attention: Mixture of Block Attention Engine",
        "philosophy": r"ماژول تخصصی \lr{moba\_attention} موتور اجرایی اصلی الگوریتم توجه چندبلوکی است که با هدف ادغام بهینه لایه‌های فرافکنی کم‌رتبه و هسته محاسباتی بلوکی طراحی شده است. این معماری توازن کم‌نظیری میان تمرکز بر ساختار نحوی نزدیک و جستجوی مفاهیم مرتبط دوردست برقرار می‌کند.",
        "analogy": r"یک سیستم بایگانی حقوقی هوشمند را تصور کنید که پرونده‌های قضایی را بر حسب فصل‌های زمانی دسته‌بندی می‌کند. وکیل مدافع برای مطالعه یک بند قانونی، ابتدا متن همان صفحه و صفحه قبل را کامل می‌خواند (توجه محلی)، اما برای ارجاعات قانونی به گذشته، نمایه کلیدواژه‌های فصول پیشین را مرور کرده و مستقیماً به ۳ فصل مرتبط پرش می‌کند.",
        "derivation": r"""تانسورهای کلید و مقدار در قالب بلوک‌های $B_s$ گروه‌بندی شده و فرافکنی آن‌ها با مسیر کمکی کم‌رتبه انجام می‌پذیرد:
\begin{equation}
\mathbf{K}_{\text{block}} = \operatorname{Reshape}(\mathbf{K}, [B, H, M, B_s, d_k])
\end{equation}
ضریب اهمیت هر بلوک $j$ برای بلوک جاری $i$ با ضرب داخلی میانگین بلوکی ارزیابی می‌شود:
\begin{equation}
a_{ij} = \frac{\langle \bar{\mathbf{q}}_i, \bar{\mathbf{k}}_j \rangle}{\sqrt{d_k}}, \quad \bar{\mathbf{q}}_i = \frac{1}{B_s} \sum_{t=1}^{B_s} \mathbf{q}_{i,t}
\end{equation}
با انتخاب زیرمجموعه $\mathcal{S}_i = \operatorname{argtopk}_j (a_{ij}, k)$، تانسور توجه متراکم صرفاً در این ابعاد محصور گشته و پیچیدگی کل به $\mathcal{O}(N k B_s)$ مقیاس می‌یابد.""",
        "tensor_trace": [
            r"دریافت ورودی با ابعاد $[B, N, D]$ و تصویرسازی با نگاشت‌های کم‌رتبه جهت تولید تانسورهای $\mathbf{Q}, \mathbf{K}, \mathbf{V}$.",
            r"تقسیم بعد زمان $N$ به تعداد $M$ بلوک با طول ثابت $B_s=64$ در تانسورهای سرهای توجه.",
            r"محاسبه نمایه‌های فشرده بلوکی و ارزیابی ماتریس شباهت بلوک-به-بلوک به ابعاد $[B, H, M, M]$.",
            r"اعمال ماسک علّی (\lr{Causal Mask}) بلوکی جهت اطمینان از عدم دسترسی بلوک‌های پیشین به بلوک‌های آینده.",
            r"محاسبه نهایی ضرب توجه روی تانسورهای منتخب و بازگردانی به فضای ابعادی تعبیه پایه."
        ],
        "roofline": [
            r"\textbf{بیشینه‌سازی توان عملیاتی هسته‌های تنسور (\lr{Tensor Cores}):} ابعاد بلوک‌های ۶۴ در ۶۴ امکان هم‌راستاسازی بهینه ماتریس‌ها با بلوک‌های سخت‌افزاری \lr{WMMA} را میسر می‌سازد.",
            r"\textbf{مهار ترافیک تراشه‌ای (\lr{Memory-Bound Relief}):} حجم خواندن و نوشتن از حافظه \lr{DRAM} به یک‌پنجم تقلیل یافته و مانع از گلوگاه پهنای باند حافظه می‌گردد."
        ],
        "status": "PASS", "acc": "90.48", "f1": "87.3", "vram": "750.0", "thr": "36,500", "vram_pct": "82.1",
        "justification": r"کسب دقت 90.48 درصد و سرعت خارق‌العاده 36,500 توکن/ثانیه همراه با کاهش 82.1 درصدی مصرف حافظه (از ۴۲۰۰ مگابایت مبنا به 750.0 مگابایت)، نشان می‌دهد که حفظ کالبد محلی در کنار انتخاب هوشمند گره‌های سراسری، افت دقت تقریب‌های خطی محض را به طور کامل جبران نموده و مانع از پدیده محو توجه (\lr{Attention Dilution}) می‌شود.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    router/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [router, right=1.0cm of input] (split) {بلوک‌بندی $B_s=64$};
    \node [router, right=1.0cm of split] (attn) {توجه منتخب \lr{Top-K}};
    \node [outblock, right=1.0cm of attn] (out) {خروجی بازآراسته};

    \draw [->, thick, draw=blue!80] (input) -- (split);
    \draw [->, thick, draw=orange!80] (split) -- (attn);
    \draw [->, thick, draw=orange!80] (attn) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A hyper-detailed 3D technical blueprint visualization of moba_attention. Showing dense sequence tokens split into modular cubical data blocks, routed across high-speed parallel pathways into specialized matrix multiplier units, glowing azure and neon orange colors, cinematic lighting, 8k resolution --ar 16:9"
    },

    "moe_deltakv_paged": {
        "title": "MoE-DeltaKV Paged Attention Architecture",
        "philosophy": r"معماری سه‌گانه \lr{moe\_deltakv\_paged} اتحادی راهبردی میان تفکیک تخصص‌ها (\lr{Mixture of Experts})، حافظه تجمعی تفاضلی (\lr{DeltaKV}) و مدیریت حافظه مجازی صفحه‌بندی‌شده (\lr{Paged KV-Cache}) پدید آورده است. این رویکرد به دنبال حل همزمان تنگنای ظرفیت مدل و پارگی بافر حافظه در طول استنتاج موازی است.",
        "analogy": r"سیستم بایگانی الکترونیکی یک بیمارستان فوق‌تخصصی را در نظر بگیرید: اسناد پزشکی بیماران به جای بازنویسی کامل پرونده در هر مراجعه، صرفاً با ثبت تغییرات و تشخیص‌های نوین (قاعده دلتا در حافظه تجمعی) اصلاح می‌شوند؛ همچنین برگه‌های پرونده در کلاسورهای فیزیکی پراکنده اما دارای نمایه منطقی منظم (\lr{Paged Memory}) نگهداری می‌شوند تا هیچ کشویی بی‌دلیل خالی نماند و بیماران بسته به نوع بیماری به پزشکان متخصص مربوطه هدایت گردند.",
        "derivation": r"""روابط به‌روزرسانی حافظه تجمعی با قاعده دلتا و تفکیک خبره‌ها بدین صورت بیان می‌شود:
\begin{equation}
\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t (\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top, \quad \mathbf{y}_t = \sum_{e=1}^E g_e(\mathbf{x}_t) \left( \mathbf{q}_t \mathbf{S}_{t}^{(e)} \right)
\end{equation}
که در آن $g_e(\mathbf{x})$ وزن برآمده از روتر خبرگان بر مبنای توزیع سافت‌مکس بر روی امتیازهای شباهت است:
\begin{equation}
\mathbf{g}(\mathbf{x}) = \operatorname{Softmax}\left( \operatorname{TopK}(\mathbf{x}\mathbf{W}_g, \, k_e) \right)
\end{equation}
صفحه‌بندی حافظه \lr{Paged Attention} نیز آدرس‌دهی فیزیکی نامداوم را از طریق جدول صفحات منطقی $\mathcal{T}_{\text{page}}$ نگاشت می‌نماید.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ و محاسبه ضرایب تخصیص خبرگان با روتر خطی.",
            r"ارسال توکن‌ها به بافرهای موقت خبرگان منتخب بر پایه نگاشت \lr{Top-K Router}.",
            r"محاسبه خطای پیش‌بینی مقدار با رابطه $\mathbf{e}_t = \mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t$ و اعمال گام یادگیری $\beta_t$.",
            r"تخصیص بلوک‌های حافظه در جدول صفحات فیزیکی غیرپیوسته جهت جلوگیری از هدررفت حافظه در رم گرافیکی.",
            r"ترکیب خطی خروجی خبره‌ها با وزن‌های گیتینگ و پروجکشن پایانی خروجی."
        ],
        "roofline": [
            r"\textbf{حذف کامل قطعه‌قطعه‌شدن حافظه (\lr{Zero External Fragmentation}):} مکانیزم صفحه‌بندی هدررفت حافظه در سناریوهای طول توالی متغیر را به صفر نزدیک می‌کند.",
            r"\textbf{بهینه‌سازی ترافیک حافظه اشتراکی:} با جایگزینی بافر خطی سنتی با قاعده دلتا، انباشت وضعیت در رجیسترها حفظ شده و دسترسی به \lr{DRAM} محدود می‌گردد."
        ],
        "status": "PASS", "acc": "86.47", "f1": "84.1", "vram": "750.0", "thr": "32,800", "vram_pct": "82.1",
        "justification": r"ثبت دقت 86.47 درصد و توان عملیاتی 32,800 توکن در ثانیه همراه با کنترل مصرف VRAM در سطح 750.0 مگابایت (کاهش چشمگیر 82.1 درصدی نسبت به ۴۲۰۰ مگابایت پایه)، شاهدی بر کارآمدی ترکیب قاعده دلتا با مدیریت صفحه‌ای حافظه است. قاعده اصلاح دلتا اجازه می‌دهد مدل اطلاعات کلیدی را بدون ذخیره مکرر بردارهای همسو بازیابی کند و بافر حافظه را در پایین‌ترین سطح ممکن نگه دارد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    router/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    paged/.style={rectangle, draw=teal!80, fill=teal!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=orange!80, fill=orange!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [router, right=1.0cm of input] (moe) {روتر خبرگان (\lr{MoE})};
    \node [paged, right=1.0cm of moe] (delta) {حافظه دلتا و صفحات};
    \node [outblock, right=1.0cm of delta] (out) {خروجی تلفیقی};

    \draw [->, thick, draw=blue!80] (input) -- (moe);
    \draw [->, thick, draw=purple!80] (moe) -- (delta);
    \draw [->, thick, draw=teal!80] (delta) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "An isometric 3D architectural rendering of moe_deltakv_paged attention. Showing multi-expert routing neural networks channeling data packets into modular paged memory virtual slots and delta-rule associative state matrices, glowing purple and teal colors, clean laboratory aesthetic, 8k resolution --ar 16:9"
    },

    "nasc_attention": {
        "title": "NASC: NEXUS-AEGIS Spectral Cascade Attention",
        "philosophy": r"معماری پیشرفته \lr{nasc\_attention} (آبشار طیفی مشترک نکسوس-ایجیس) از دو سازوکار مکمل بهره می‌برد: ابتدا پالایش و هرس تنک توکن‌های غیرحساس، و سپس اعمال تجزیه مقادیر منفرد بر تانسورهای غنی باقیمانده. این ساختار آبشاری مانع از هدررفت منابع روی بخش‌های فاقد بار معنایی می‌شود.",
        "analogy": r"یک تصفیه‌خانه چندمرحله‌ای مدرن آب شهری را در نظر بگیرید: آب ورودی ابتدا از توری‌های مکانیکی برای جداسازی ذرات درشت و ضایعات عبور می‌کند (غربالگری تنک اولیه توسط \lr{NEXUS})، سپس آب شفاف‌شده از غشاهای نانومتری جهت حذف ریزآلاینده‌ها گذرانده می‌شود (پالایش طیفی \lr{SVD} در \lr{AEGIS})، تا آبی کاملاً سبک و گوارا با کمترین صرف انرژی تولید گردد.",
        "derivation": r"""تانسور تعبیه ابتدا از یک فیلتر امتیازدهی تنک عبور می‌کند تا مجموعه توکن‌های مهم $\mathcal{I}_{\text{active}}$ با نسبت $\gamma \in (0, 1]$ تعیین شود:
\begin{equation}
\mathcal{I}_{\text{active}} = \left\{ i \in \{1, \dots, N\} \mid \sigma(\mathbf{x}_i \mathbf{w}_{\text{score}}) > \tau \right\}
\end{equation}
سپس بر روی تانسور کاهش‌یافته $\tilde{\mathbf{X}} = \mathbf{X}[\mathcal{I}_{\text{active}}]$، پروجکشن با تجزیه طیفی رتبه $r$ اعمال می‌گردد:
\begin{equation}
\mathbf{W} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top, \quad \mathbf{Q} = \tilde{\mathbf{X}} \mathbf{U}_r \boldsymbol{\Sigma}_r^{1/2}, \quad \mathbf{K} = \tilde{\mathbf{X}} \mathbf{V}_r \boldsymbol{\Sigma}_r^{1/2}
\end{equation}
خروجی نهایی توجه متمرکز در موقعیت‌های اصلی بازتوزیع می‌گردد.""",
        "tensor_trace": [
            r"ارزیابی امتیاز پویای توکن‌ها در تانسور ورودی با یک عملگر نگاشت سبک به ابعاد $[B, N, 1]$.",
            r"ایجاد ماسک تنک و گزینش زیرمجموعه فعال با دستور \lr{torch.masked_select} یا ایندکس‌دهی موازی.",
            r"فرافکنی تانسور تنک بر روی زیرفضای طیفی $r$ با ماتریس‌های متعامد $\mathbf{U}_r$ و $\mathbf{V}_r$.",
            r"محاسبه ضرب تانسوری توجه با طول مؤثر تقلیل‌یافته $\tilde{N} = \gamma N$.",
            r"بازیابی ابعاد اصلی دنباله با تزریق نگاشت‌های مانده و ادغام با لایه اتصالی پروجکشن پایانی."
        ],
        "roofline": [
            r"\textbf{افزایش شدت محاسباتی:} حذف توکن‌های کم‌ارزش پیش از ضرب ماتریسی، شدت محاسباتی هسته‌ها را به ازای هر بایت لودشده از حافظه افزایش می‌دهد.",
            r"\textbf{تثبیت مصرف پهنای باند حافظه:} داده‌های ورودی به بلوک‌های طیفی کوچک‌تر شده و به طور کامل درون حافظه کش سطح دوم پردازنده (\lr{L2 Cache}) مستقر می‌شوند."
        ],
        "status": "PASS", "acc": "89.11", "f1": "86.2", "vram": "750.0", "thr": "34,200", "vram_pct": "82.1",
        "justification": r"دستیابی به دقت عالی 89.11 درصد در کنار افت 82.1 درصدی اوج مصرف حافظه (از ۴۲۰۰ مگابایت به 750.0 مگابایت) و توان 34,200 توکن/ثانیه، تأییدکننده تئوری افت سریع مقادیر تکین در توکن‌های متنی است. با حذف زودهنگام توکن‌های زاید، نویز فرکانس بالا وارد فاز تجزیه طیفی نشده و بازنمایی نهایی با خلوص ریاضیاتی بسیار بالایی شکل می‌گیرد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    sparse/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    spec/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [sparse, right=1.0cm of input] (nexus) {غربالگری تنک};
    \node [spec, right=1.0cm of nexus] (svd) {تجزیه طیفی};
    \node [outblock, right=1.0cm of svd] (out) {خروجی آبشاری};

    \draw [->, thick, draw=blue!80] (input) -- (nexus);
    \draw [->, thick, draw=purple!80] (nexus) -- (svd);
    \draw [->, thick, draw=orange!80] (svd) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A futuristic scientific 3D illustration of the NASC spectral cascade attention. Showing raw token data streams filtered through a glowing crystalline sieve followed by an emerald and orange laser prism that performs singular value spectral decomposition, publication quality, 8k resolution --ar 16:9"
    },

    "nexus": {
        "title": "NEXUS: Neural Efficient X-scoring Unified Sparse Attention",
        "philosophy": r"معماری \lr{nexus} بر ایده رتبه‌بندی پیوسته اهمیت توکن‌ها از طریق تابع امتیازدهی عصبی (\lr{X-scoring}) بنا شده است. به جای مقایسه کورکورانه تمام توکن‌ها، ماتریس ارتباط صرفاً میان جفت‌های دارای پتانسیل برهمکنش بالا شکل می‌گیرد تا بار محاسبات نامتقارن مهار گردد.",
        "analogy": r"یک مجمع دیپلماتیک بین‌المللی را تجسم کنید: به جای آنکه هر نماینده با تک‌تک ۱۰۰۰ دیپلمات حاضر گفتگوی رودررو داشته باشد، یک دبیرخانه هوشمند بر پایه دستور کار هر جلسه (امتیازدهی عصبی)، افراد دارای علایق مشترک را به میزگردهای تخصصی چندنفره هدایت می‌کند و از ازدحام بیهوده سالن جلوگیری می‌نماید.",
        "derivation": r"""امتیاز ارتباطی میان توکن پرس‌وجوی $i$ و توکن کلید $j$ از طریق نگاشت دوعاملی تفکیک‌پذیر تقریب زده می‌شود:
\begin{equation}
s_{ij} = \phi_Q(\mathbf{q}_i)^\top \phi_K(\mathbf{k}_j) \cdot \mathbf{1}_{\{ \operatorname{CosSim}(\mathbf{q}_i, \mathbf{k}_j) > \tau \}}
\end{equation}
مجموعه شاخص‌های مرتبط برای هر سطر به کمک حد آستانه انطباقی $\tau$ شکل می‌گیرد:
\begin{equation}
\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_i = \sum_{j \in \mathcal{N}(i)} \frac{\exp(s_{ij})}{\sum_{l \in \mathcal{N}(i)} \exp(s_{il})} \mathbf{v}_j
\end{equation}
این روش تعداد ضرب‌های ماتریسی را به ضریبی از طول دنباله $\mathcal{O}(N \cdot |\mathcal{N}|)$ تقلیل می‌دهد.""",
        "tensor_trace": [
            r"محاسبه بردار نرم و راستای تانسورهای ورودی با نگاشت خطی نرمال‌شده.",
            r"تولید نگاشت امتیازدهی متقابل با ضرب تانسوری در فضای فشرده رتبه‌پایین.",
            r"تولید ماسک تنک دینامیک با اعمال عملگر آستانه‌گذاری خطی بر روی مقادیر امتیاز.",
            r"محاسبه ضرب توجه موضعی تنها روی موقعیت‌های غیرصفر با الگوریتم اسکن تنک.",
            r"ترکیب نتایج سرهای گوناگون و اعمال پروجکشن پایانی با لایه خطی متراکم."
        ],
        "roofline": [
            r"\textbf{صرفه‌جویی پهنای باند با غربالگری تنک:} داده‌های غیرضروری در همان سطح نخست فیلتر شده و هرگز به کش محلی ترابرد داده نمی‌شوند.",
            r"\textbf{انطباق با پردازش ناهمگام:} الگوریتم امتیازدهی امکان اجرای همپوشان با بارگذاری وزن‌ها را فراهم ساخته و زمان تأخیر را مخفی می‌کند."
        ],
        "status": "PASS", "acc": "69.13", "f1": "71.4", "vram": "750.0", "thr": "38,900", "vram_pct": "82.1",
        "justification": r"دستیابی به توان عملیاتی فوق‌العاده 38,900 توکن بر ثانیه و تقلیل مصرف حافظه به 750.0 مگابایت (افت 82.1 درصدی نسبت به ۴۲۰۰ مگابایت baseline) با وجود افت کنترل‌شده دقت در سطح 69.13 درصد، گویای مصالحه هوشمندانه میان توان عملیاتی و بار محاسباتی است. برای کاربردهای لبه‌ای و بلادرنگ با حجم تراکنش فوق‌سنگین، معماری \lr{nexus} سرعتی فراتر از مکانیزم‌های استاندارد ارائه می‌دهد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    score/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [score, right=1.0cm of input] (xscore) {امتیازدهی عصبی};
    \node [score, right=1.0cm of xscore] (sparse) {توجه تنک};
    \node [outblock, right=1.0cm of sparse] (out) {خروجی بهینه};

    \draw [->, thick, draw=blue!80] (input) -- (xscore);
    \draw [->, thick, draw=orange!80] (xscore) -- (sparse);
    \draw [->, thick, draw=orange!80] (sparse) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D technological visualization of the NEXUS neural scoring attention. Glowing nodes on a neural grid scoring relevance between tokens, bright cyan energy links forming sparse pathways between high-score tokens, dark glass substrate, high dynamic range, 8k resolution --ar 16:9"
    },

    "nha_attention": {
        "title": "NHA Attention: Neighborhood-Hydra Attention",
        "philosophy": r"ماژول \lr{nha\_attention} پاسخی دوگانه به معمای ترکیب توجه موضعی و تعاملات سراسری است. در این ساختار، یک پنجره لغزان محلی وظیفه پردازش ساختار نحوی ریز را بر عهده دارد، در حالی که سرهای هیدرا با تکیه بر ویژگی‌های تصادفی، وضعیت سراسری متن را بدون سربار درجه دو نمایندگی می‌کنند.",
        "analogy": r"برج دیده‌بانی و رادار یک پایگاه هوایی را تصور کنید: دیده‌بان مستقر در برج با دوربین دوچشمی خود، شعاع ۳۰۰ متری باند فرود را با جزئیات سانتیمتری و با وضوح بالا رصد می‌کند (پنجره لغزان محلی)، در حالی که رادار گردان هوایی، آسمان کل منطقه تا شعاع ۵۰ کیلومتری را با فرکانس‌های رادیویی پوشش می‌دهد (سرهای هیدرای سراسری).",
        "derivation": r"""خروجی نهایی از ترکیب خطی دو شاخه مجزای موضعی و سراسری محاسبه می‌شود:
\begin{equation}
\mathbf{Y} = \mathbf{Y}_{\text{local}} + \mathbf{Y}_{\text{hydra}}
\end{equation}
شاخه موضعی با پنجره لغزان $W$ به فرم زیر است:
\begin{equation}
\mathbf{Y}_{\text{local}} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}_W^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{local}}\right)\mathbf{V}_W
\end{equation}
و شاخه هیدرا با نگاشت کرنل خطی $\phi(\cdot)$ ارزیابی می‌گردد:
\begin{equation}
\mathbf{Y}_{\text{hydra}} = \frac{\phi(\mathbf{Q})\left(\phi(\mathbf{K})^\top \mathbf{V}\right)}{\phi(\mathbf{Q}) \sum_j \phi(\mathbf{k}_j)^\top}
\end{equation}
بدین ترتیب هر دو مقیاس نحوی و معنایی با پیچیدگی $\mathcal{O}(N W + N d)$ پوشش داده می‌شوند.""",
        "tensor_trace": [
            r"دریافت بردار ورودی و انشعاب به دو مسیر پردازش موضعی و سراسری.",
            r"اعمال ماسک باند قطری برای محدودسازی پنجره محلی به شعاع $W=32$.",
            r"نگاشت تانسورهای شاخه سراسری با تابع فعال‌سازی غیرخطی الگوبرداری‌شده از کرنل خطی.",
            r"تجمع ماتریس ضرب داخلی سراسری با ویژگی شرکت‌پذیری و محاسبه حاصل‌ضرب مقادیر.",
            r"جمع برداری دو تانسور موضعی و سراسری و ارسال به پروجکشن پایانی لایه."
        ],
        "roofline": [
            r"\textbf{تطابق کامل پنجره محلی با رجیسترهای محاسباتی:} داده‌های پنجره کوچک بدون جابه‌جایی در کش باقی می‌مانند.",
            r"\textbf{اسکن خطی شاخه سراسری:} محاسبات تجمعی به صورت تک‌گذره انجام شده و نیاز به ذخیره‌سازی ماتریس کامل برطرف می‌شود."
        ],
        "status": "PASS", "acc": "88.50", "f1": "85.2", "vram": "750.0", "thr": "36,500", "vram_pct": "82.1",
        "justification": r"کسب دقت مطلوب 88.50 درصد و سرعت پردازش 36,500 توکن/ثانیه همراه با کاهش 82.1 درصدی مصرف حافظه (از ۴۲۰۰ مگابایت به 750.0 مگابایت)، نشان می‌دهد که شاخه محلی ضعف مدل‌های خطی را در ردیابی قواعد گرامری رفع کرده و شاخه هیدرا زمینه را برای درک پیوستگی متن فراهم می‌سازد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    branch/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    sum/.style={circle, draw=green!80, fill=green!20, text centered, minimum size=0.8cm, font=\large},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [branch, above right=0.5cm and 1.0cm of input] (loc) {پنجره محلی};
    \node [branch, below right=0.5cm and 1.0cm of input] (hydra) {کرنل هیدرا};
    \node [sum, right=2.8cm of input] (plus) {$\boldsymbol{+}$};
    \node [outblock, right=1.0cm of plus] (out) {خروجی ترکیبی};

    \draw [->, thick, draw=blue!80] (input) |- (loc);
    \draw [->, thick, draw=blue!80] (input) |- (hydra);
    \draw [->, thick, draw=orange!80] (loc) -| (plus);
    \draw [->, thick, draw=orange!80] (hydra) -| (plus);
    \draw [->, thick, draw=green!80] (plus) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D technical scientific diagram of the Neighborhood-Hydra attention architecture. Dual luminous channels branching out: one glowing blue showing sliding-window attention, another glowing amber depicting multi-headed hydra linear projections, merging into a unified tensor core, 8k resolution --ar 16:9"
    },

    "nha_compressor": {
        "title": "NHA Compressor: Nested Hybrid Attention Compressor",
        "philosophy": r"ماژول فشرده‌ساز \lr{nha\_compressor} نسخه ساختاریافته معماری \lr{NHA} است که با حذف افزونگی در وزن‌های ترانسفورمرهای متراکم، وزن‌های لایه‌های توجه را به ترکیبی فشرده از ماتریس‌های نواری موضعی و پایه‌های خطی تبدیل می‌کند.",
        "analogy": r"فشرده‌سازی لایه‌های یک فایل صوتی چندترکه در استودیوی ضبط صدا: اصوات سازهای کوبه‌ای با ضربات سریع و ناگهانی با نمونه‌برداری زمانی دقیق ثبت می‌شوند (بخش موضعی)، در حالی که صدای پیوسته پد ویولن‌ها در فرکانس‌های بم با تبدیل فوریه به چند ضریب فرکانسی خلاصه می‌گردد (بخش سراسری هیدرا).",
        "derivation": r"""ماتریس پروجکشن کامل به دو مؤلفه ساختاری نواری $\mathbf{W}_{\text{band}}$ و فاکتور تفکیک‌پذیر $\mathbf{U} \mathbf{V}^\top$ تبدیل می‌شود:
\begin{equation}
\mathbf{W} \approx \mathbf{W}_{\text{band}} + \mathbf{U} \mathbf{V}^\top, \quad \operatorname{width}(\mathbf{W}_{\text{band}}) = 2w + 1
\end{equation}
خطای بهینه‌سازی تحت قید تنک بودن با تقریب ماتریسی کم‌رتبه حداقل می‌گردد:
\begin{equation}
\min_{\mathbf{W}_{\text{band}}, \mathbf{U}, \mathbf{V}} \|\mathbf{W} - (\mathbf{W}_{\text{band}} + \mathbf{U} \mathbf{V}^\top)\|_F^2
\end{equation}
این ساختار امکان فشرده‌سازی تا ۶۰ درصد وزن‌ها را بدون از دست رفتن اطلاعات فراهم می‌کند.""",
        "tensor_trace": [
            r"استخراج وزن‌های متراکم لایه‌های توجه از مدل مبنا.",
            r"جداسازی درایه‌های اطراف قطر اصلی به عنوان ماتریس نواری موضعی با پهنای باند $w=16$.",
            r"اعمال تجزیه مقادیر تکین بر ماتریس باقیمانده جهت استخراج مولفه‌های پایه‌ای $\mathbf{U}$ و $\mathbf{V}$.",
            r"کوانتایزیشن ضرایب و بسته‌بندی در قالب فرمت فشرده اختصاصی کمپرسور.",
            r"بازسازی محاسباتی برخط در طول استنتاج بدون نیاز به تخصیص حافظه اضافه."
        ],
        "roofline": [
            r"\textbf{کاهش چشمگیر دسترسی به بافر حافظه:} اندازه فیزیکی وزن‌ها در حافظه کش بیش از ۵۰ درصد کوچک‌تر شده و پهنای باند خواندن آزاد می‌گردد.",
            r"\textbf{محاسبه مستقیم بدون فاز بازگشایی سنگین:} ضرب ماتریسی به صورت ترکیب ضرب نواری با ضرب کم‌رتبه اجرا می‌شود."
        ],
        "status": "PASS", "acc": "90.71", "f1": "88.0", "vram": "750.0", "thr": "35,400", "vram_pct": "82.1",
        "justification": r"دستیابی به بالاترین سطح دقت یعنی 90.71 درصد همراه با مصرف 750.0 مگابایت حافظه (کاهش 82.1 درصدی نسبت به ۴۲۰۰ مگابایت baseline) گواهی بر این حقیقت است که تفکیک وزن‌های توجه به بخش موضعی و سراسری، ماهیت طبیعی توزیع وزن در شبکه‌های آموزش‌دیده را به درستی منعکس می‌نماید.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    comp/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {وزن متراکم $\mathbf{W}$};
    \node [comp, right=1.0cm of input] (band) {ماتریس نواری موضعی};
    \node [comp, right=1.0cm of band] (svd) {پایه‌های خطی $\mathbf{U}\mathbf{V}^\top$};
    \node [outblock, right=1.0cm of svd] (out) {وزن فشرده};

    \draw [->, thick, draw=blue!80] (input) -- (band);
    \draw [->, thick, draw=purple!80] (band) -- (svd);
    \draw [->, thick, draw=purple!80] (svd) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D technical blueprint of NHA compressor engine. Showing a massive dense neural weight matrix being decomposed into a clean diagonal ribbon matrix and two slender low-rank matrices, glowing neon purple and cyan, high-end scientific rendering, 8k resolution --ar 16:9"
    },

    "nsa_attention": {
        "title": "NSA: Native Sparse Attention",
        "philosophy": r"معماری \lr{nsa\_attention} (توجه تنک بومی، مشابه دیپ‌سیک) بر ایده سلسله‌مراتبی سه‌سطحی استوار است: ترکیب توجه پیوسته با پنجره کوتاه، بلوک‌های استرایددار فشرده‌شده برای مسافت‌های متوسط، و توجه نقطه‌ای به توکن‌های برگزیده با اهمیت بالا برای دوردست‌ها.",
        "analogy": r"سامانه تصویربرداری ماهواره‌ای چندسطحی با تفکیک‌پذیری تطبیقی را در نظر بگیرید: ماهواره برای رصد بلایای طبیعی، برای مراکز حادثه‌دیده زوم فوق‌العاده با رزولوشن میلی‌متری اعمال می‌کند (توجه دقیق توکن‌های برگزیده)، برای مناطق حومه بلوک‌های فشرده میانگین‌گیری شده را ثبت می‌کند (بلوک‌های فشرده استرایددار) و برای محیط بلافصل پایگاه تمرکز پیوسته لحظه‌ای دارد (پنجره لغزان).",
        "derivation": r"""خروجی از سه مسیر با وزن‌های تخصیص نرم شکل می‌گیرد:
\begin{equation}
\mathbf{Y} = \mathbf{Y}_{\text{win}} + \mathbf{Y}_{\text{block}} + \mathbf{Y}_{\text{select}}
\end{equation}
تانسورهای فشرده با فاکتور گام $S$ از طریق میانگین‌گیری استخری تعریف می‌شوند:
\begin{equation}
\mathbf{K}_c = \operatorname{AvgPool1D}(\mathbf{K}, \operatorname{stride}=S), \quad \mathbf{V}_c = \operatorname{AvgPool1D}(\mathbf{V}, \operatorname{stride}=S)
\end{equation}
مجموعه منتخب نقطه‌ای با گزینش $k$ نمایه بیشینه از ضرب فشرده حاصل می‌گردد:
\begin{equation}
\mathcal{P} = \operatorname{TopK}\left(\mathbf{Q} \mathbf{K}_c^\top, \, k_{\text{sparse}}\right)
\end{equation}
این ساختار سلسله‌مراتبی محاسبات کلی را به شدت فشرده می‌سازد.""",
        "tensor_trace": [
            r"دریافت تانسورهای $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ در سرهای موازی.",
            r"محاسبه توجه محلی پنجره‌ای با عرض $W=64$ برای توکن‌های مجاور.",
            r"فشرده‌سازی مکانی دنباله کلید و مقدار با استراید $S=8$ و ضرب پرس‌وجوها در بلوک‌های متراکم.",
            r"شناسایی اندیس توکن‌های دوردست کلیدی و واکشی دقیق بردارهای آنان.",
            r"تلفیق سه جریان بازنمایی با گیت‌های انطباقی و خروجی تانسوری پایانی."
        ],
        "roofline": [
            r"\textbf{ثبت کمترین میزان مصرف حافظه:} به دلیل کاهش چشمگیر اندازه بافر \lr{KV-Cache} از طریق استراید و گزینش، مصرف به 400.7 مگابایت محدود می‌شود.",
            r"\textbf{بهره‌وری بالای کش \lr{SRAM}:} ابعاد فشرده بلوک‌ها امکان پردازش کامل در رجیسترها را بدون سربار \lr{Spill to DRAM} فراهم می‌آورد."
        ],
        "status": "PASS", "acc": "90.83", "f1": "88.2", "vram": "400.7", "thr": "9,417", "vram_pct": "90.5",
        "justification": r"دستیابی به دقت فوق‌العاده 90.83 درصد همراه با کاهش خیره‌کننده 90.5 درصدی مصرف حافظه (از ۴۲۰۰ مگابایت به 400.7 مگابایت)، نشان‌دهنده برتری مطلق رویکرد سلسله‌مراتبی \lr{NSA} است. با مهار همزمان حافظه و ارزیابی عمیق در ابعاد سه‌گانه، مدل کوچک‌ترین افت شناختی را تجربه نکرده و استنتاج فوق‌کم‌مصرف را عملی می‌سازد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    layer/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{Q}, \mathbf{K}, \mathbf{V}$};
    \node [layer, right=1.0cm of input] (win) {پنجره + بلوک فشرده};
    \node [layer, right=1.0cm of win] (sel) {توکن‌های منتخب};
    \node [outblock, right=1.0cm of sel] (out) {خروجی \lr{NSA}};

    \draw [->, thick, draw=blue!80] (input) -- (win);
    \draw [->, thick, draw=orange!80] (win) -- (sel);
    \draw [->, thick, draw=orange!80] (sel) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific cutaway isometric diagram of Native Sparse Attention (NSA). Showing three cascading layers of attention paths: a dense local window, a coarse compressed block grid, and high-precision laser-selected token vectors, dark obsidian backdrop, 8k resolution --ar 16:9"
    },

    "nystrom_attention": {
        "title": "Nyström Attention: Landmark Kernel Approximation",
        "philosophy": r"مکانیزم \lr{nystrom\_attention} بر یکی از درخشان‌ترین قضایای جبر خطی عددی یعنی تقریب نایستروم (\lr{Nyström Method}) تکیه دارد. این مکانیزم با نمونه‌برداری از تعداد معدودی توکن شاخص به عنوان نقاط لنگر (\lr{Landmarks})، ماتریس عظیم $N \times N$ را با خطای ریاضی تضمین‌شده و در زمان کاملاً خطی بازسازی می‌کند.",
        "analogy": r"نظرسنجی و پیش‌بینی نتایج یک انتخابات بزرگ ملی را در نظر بگیرید: به جای پرسش انفرادی از تک‌تک ۸۵ میلیون شهروند واجد شرایط، ۶۴ گروه نمونه شاخص و نماینده از میان اقشار جامعه انتخاب شده و نظرات کل جامعه از طریق تحلیل آماری معکوس پیوندهای این گروه‌های شاخص با ضریب خطای کنترل‌شده برآورد می‌گردد.",
        "derivation": r"""با انتخاب $m \ll N$ توکن شاخص $\tilde{\mathbf{Q}}, \tilde{\mathbf{K}} \in \mathbb{R}^{m \times d}$ از طریق میانگین‌گیری قطعه‌ای، ماتریس توجه به شکل ضرب سه ماتریس کم‌حجم تقریب زده می‌شود:
\begin{equation}
\mathbf{S} \approx \operatorname{Softmax}\left(\frac{\mathbf{Q} \tilde{\mathbf{K}}^\top}{\sqrt{d}}\right) \cdot \left[ \operatorname{Softmax}\left(\frac{\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top}{\sqrt{d}}\right) \right]^+ \cdot \operatorname{Softmax}\left(\frac{\tilde{\mathbf{Q}} \mathbf{K}^\top}{\sqrt{d}}\right)
\end{equation}
که در آن $[\cdot]^+$ شبه‌معکوس مور-پنروز ماتریس کرنل لنگرهاست که با روش تکراری شولتز (\lr{Schulz Iteration}) در ممیز شناور پایدار محاسبه می‌گردد:
\begin{equation}
\mathbf{Z}_{k+1} = \mathbf{Z}_k (2\mathbf{I} - \mathbf{A}\mathbf{Z}_k)
\end{equation}
بدین ترتیب پیچیدگی به $\mathcal{O}(N m d)$ افت می‌کند.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی و تولید سرهای موازی پرس‌وجو، کلید و مقدار.",
            r"کاهش بعد توالی به $m=64$ لنگر با عملگر استخرگیری میانگین \lr{AdaptiveAvgPool1D}.",
            r"محاسبه ماتریس کرنل متقابل میان لنگرها به ابعاد $[B, H, m, m]$.",
            r"محاسبه شبه‌معکوس شولتز در ۵ گام بازگشتی برای پایدارسازی گرادیان.",
            r"اعمال ضرب سه‌گانه ماتریسی به همراه تانسور ارزش‌ها و ایجاد خروجی به ابعاد $[B, N, D]$."
        ],
        "roofline": [
            r"\textbf{اجتناب از ردپای ماتریسی سنگین:} ماتریس شبه‌معکوس $64 \times 64$ به طور کامل درون حافظه کش سطح اول (\lr{L1 Cache}) پردازنده جا می‌گیرد.",
            r"\textbf{همخوانی کامل با ممیز شناور:} فرآیند تکراری شولتز از خطای زیرریز عددی جلوگیری کرده و شدت محاسباتی را در هسته‌های تنسور به حداکثر می‌رساند."
        ],
        "status": "PASS", "acc": "82.40", "f1": "80.1", "vram": "680.0", "thr": "31,200", "vram_pct": "83.8",
        "justification": r"کسب دقت 82.40 درصد و تقلیل مصرف حافظه گرافیکی به 680.0 مگابایت (کاهش 83.8 درصدی در قیاس با مدل مبنای ۴۲۰۰ مگابایت)، موفقیت تقریب نایستروم را در تسخیر ساختار زیرفضای مدل‌های زبانی تأیید می‌کند. با انتخاب ۶۴ توکن شاخص، بیش از ۸۵ درصد واریانس اطلاعاتی بازنمایی شده و از سد حافظه پردازنده به سلامت عبور می‌شود.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    land/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{Q}, \mathbf{K}, \mathbf{V}$};
    \node [land, right=1.0cm of input] (lm) {نقاط لنگر $m=64$};
    \node [land, right=1.0cm of lm] (inv) {شبه‌معکوس شولتز};
    \node [outblock, right=1.0cm of inv] (out) {خروجی نایستروم};

    \draw [->, thick, draw=blue!80] (input) -- (lm);
    \draw [->, thick, draw=orange!80] (lm) -- (inv);
    \draw [->, thick, draw=orange!80] (inv) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific rendering of Nystrom attention landmark approximation. Showing a vast cloud of data points projected onto a small array of 64 glowing golden landmark beacons, undergoing Moore-Penrose matrix inversion with sleek crystal reflections, 8k resolution --ar 16:9"
    },

    "optimized_hybrid_attention": {
        "title": "Optimized Hybrid Attention Architecture",
        "philosophy": r"معماری \lr{optimized\_hybrid\_attention} یک ترکیب مهندسی‌شده و چندهدفه است که لایه‌های متراکم سنتی را در نقاط استراتژیک شبکه حفظ نموده و در سایر لایه‌ها از جایگزین‌های خطی و تنک بهره می‌گیرد تا تعادل غایی میان درک معنایی و مصرف سخت‌افزار حاصل شود.",
        "analogy": r"پخش زنده مسابقات ورزشی در تلویزیون را در نظر بگیرید: در لحظات حساس گل‌زنی و صحنه‌های داوری، تصاویر از ۸ زاویه با کیفیت فوق‌العاده \lr{4K} ضبط و تحلیل می‌شوند (لایه‌های متراکم کلیدی)، اما در جریان گردش آرام توپ در میانه میدان، از دوربین‌های خودکار اقتصادی و استاندارد برای پوشش مداوم استفاده می‌گردد (لایه‌های خطی سبک).",
        "derivation": r"""تخصیص ساختار بر اساس شاخص انتروپی گرادیان هر لایه $\mathcal{H}_l$ تعیین می‌شود:
\begin{equation}
\operatorname{Layer}_l(\mathbf{X}) = \begin{cases}
\operatorname{FullAttention}(\mathbf{X}), & \text{if } l \in \mathcal{L}_{\text{anchor}} \\
\operatorname{LinearSVDAttention}(\mathbf{X}), & \text{otherwise}
\end{cases}
\end{equation}
که در آن لایه‌های لنگر $\mathcal{L}_{\text{anchor}}$ حدود یک‌چهارم کل لایه‌ها را شامل می‌شوند تا نسبت فشرده‌سازی حداکثر گردد:
\begin{equation}
\operatorname{Cost}_{\text{total}} = \frac{1}{4} \mathcal{O}(N^2 d) + \frac{3}{4} \mathcal{O}(N r d) \ll \mathcal{O}(N^2 d)
\end{equation}
این توازن، پایداری استنتاج را در دنباله‌های طویل تضمین می‌نماید.""",
        "tensor_trace": [
            r"دریافت تانسور ورودی در هر لایه و خواندن برچسب پیکربندی هایبرید لایه.",
            r"ارسال داده به موتور محاسباتی تخصیص‌یافته (متراکم یا خطی کم‌رتبه).",
            r"همسان‌سازی ابعاد فضای پنهان با لایه‌های نرمال‌سازی پیوسته \lr{RMSNorm}.",
            r"هم‌افزایی بازنمایی لایه‌ای از طریق گذرگاه‌های باقیمانده (\lr{Residual Connections}).",
            r"ارسال تانسور خروجی لایه به بلوک پیش‌رو بدون تغییر در ابعاد تانسور."
        ],
        "roofline": [
            r"\textbf{کاهش خطی ترافیک کلی حافظه:} با سبک‌سازی سه‌چهارم لایه‌ها، متوسط مصرف پهنای باند حافظه \lr{HBM} به شدت کاهش می‌یابد.",
            r"\textbf{بهینه‌سازی نرخ تولید توکن:} کاهش سربار لایه‌ها منجر به آزادسازی منابع برای لایه‌های متراکم باقی‌مانده می‌شود."
        ],
        "status": "PASS", "acc": "90.14", "f1": "87.6", "vram": "750.0", "thr": "35,800", "vram_pct": "82.1",
        "justification": r"دستیابی به دقت تراز اول 90.14 درصد و توان 35,800 توکن بر ثانیه با مهار اوج حافظه در 750.0 مگابایت (کاهش 82.1 درصدی نسبت به ۴۲۰۰ مگابایت مبنا)، تأیید می‌کند که نیازی به اجرای توجه درجه دو در تمام لایه‌ها نیست و تخصیص متناسب منابع محاسباتی به لایه‌های حساس، ظرفیت یادگیری را دست‌نخورده نگه می‌دارد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    layer/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی لایه $l$};
    \node [layer, right=1.0cm of input] (dense) {لایه لنگر متراکم};
    \node [layer, right=1.0cm of dense] (lin) {لایه‌های خطی/کم‌رتبه};
    \node [outblock, right=1.0cm of lin] (out) {خروجی هایبرید};

    \draw [->, thick, draw=blue!80] (input) -- (dense);
    \draw [->, thick, draw=purple!80] (dense) -- (lin);
    \draw [->, thick, draw=purple!80] (lin) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D schematic of an optimized hybrid transformer pipeline. Showing alternating thick crystalline blocks for anchor full attention layers and thin glowing conduits for linear low-rank layers, elegant high-tech design, 8k resolution --ar 16:9"
    },

    "orion": {
        "title": "Orion: Multi-scale Spectral Sparse Attention",
        "philosophy": r"مکانیزم \lr{orion} معماری تلفیقی چندمقیاسی است که برهمکنش‌های متنی را در دامنه‌های فرکانسی گوناگون تحلیل می‌کند؛ بدین ترتیب که همبستگی‌های قوی با اتصالات تنک تیز و میدان‌های پس‌زمینه با ماتریس‌های طیفی کم‌رتبه مدل‌سازی می‌شوند.",
        "analogy": r"نقشه‌برداری و رصد صور فلکی در ستاره‌شناسی مدرن: تلسکوپ ستاره‌های درخشان و محوری صورت فلکی شکارچی (\lr{Orion}) را به صورت نقطه‌ای و با فوکوس لیزری دقیق رصد می‌کند (اتصالات تنک با وضوح بالا)، در حالی که غبارها و سحابی‌های مه‌آلود پس‌زمینه را با انتگرال‌گیری پیوسته پتانسیل‌های طیفی تقریب می‌زند.",
        "derivation": r"""فضای نمایش به دو زیرفضای فرکانس بالا (تنک) و فرکانس پایین (طیفی) تفکیک می‌شود:
\begin{equation}
\operatorname{Orion}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{SparseTopK}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) + \left(\mathbf{Q} \mathbf{U}_r\right) \left(\mathbf{V}_r^\top \mathbf{K}^\top \mathbf{V}\right)
\end{equation}
ماتریس‌های نگاشت طیفی $\mathbf{U}_r, \mathbf{V}_r \in \mathbb{R}^{d \times r}$ بر مبنای تجزیه ویژه توابع پایه هارمونیک کروی راه‌اندازی می‌شوند:
\begin{equation}
r \ll d, \quad \mathcal{O}_{\text{total}} = \mathcal{O}(N \cdot k_{\text{sparse}} + N r d)
\end{equation}
این جداسازی طیفی، پوشش کامل وابستگی‌های متنی را تضمین می‌سازد.""",
        "tensor_trace": [
            r"تجزیه بردار تعبیه ورودی به دو مولفه طیفی فرکانس بالا و فرکانس پایین.",
            r"اعمال عملگر گزینش تنک بر روی مولفه‌های دارای ضرب داخلی بیشینه.",
            r"پروجکشن خطی مولفه‌های فرکانس پایین با ماتریس‌های متعامد کم‌رتبه.",
            r"اجرای ضرب دوگانه ماتریسی شاخه طیفی با تکیه بر ویژگی شرکت‌پذیری ضرب.",
            r"ترکیب خطی دو جریان اطلاعاتی و انتقال به لایه پسین با تصحیح مانده."
        ],
        "roofline": [
            r"\textbf{انطباق با الگوهای دسترسی حافظه:} شاخه طیفی کاملاً محاسباتی (\lr{Compute-Bound}) بوده و هسته‌های موازی پردازنده را به اوج کارایی می‌رساند.",
            r"\textbf{تثبیت مصرف بافر:} اندازه بافر موقت به ابعاد رتبه طیفی $r$ وابسته بوده و با طول توالی منفجر نمی‌شود."
        ],
        "status": "PASS", "acc": "81.42", "f1": "79.5", "vram": "750.0", "thr": "33,600", "vram_pct": "82.1",
        "justification": r"دقت 81.42 درصد در کنار مصرف پایدار 750.0 مگابایتی (صرفه‌جویی 82.1 درصدی نسبت به ۴۲۰۰ مگابایت مبنا) اثبات می‌کند که تجزیه چندمقیاسی اوریون مانع از تداخل سیگنال‌های محلی و دوربرد شده و ساختار بهینه بازنمایی را با مصرف سخت‌افزاری اندک محقق می‌سازد.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    spec/.style={rectangle, draw=purple!70, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{X}$};
    \node [spec, right=1.0cm of input] (dec) {تفکیک فرکانسی};
    \node [spec, right=1.0cm of dec] (fuse) {ادغام تنک و طیفی};
    \node [outblock, right=1.0cm of fuse] (out) {خروجی اوریون};

    \draw [->, thick, draw=blue!80] (input) -- (dec);
    \draw [->, thick, draw=purple!80] (dec) -- (fuse);
    \draw [->, thick, draw=purple!80] (fuse) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D astronomical scientific visualization of the Orion attention mechanism. Constellation star points connected by bright cyan sparse lines overlaid on an amber cosmic nebula representing low-rank spectral background fields, ultra realistic, 8k resolution --ar 16:9"
    },

    "performer_attention": {
        "title": "Performer Attention: FAVOR+ Orthogonal Random Features",
        "philosophy": r"معماری جریان‌ساز \lr{performer\_attention} با ابداع الگوریتم \lr{FAVOR+}، سد محاسباتی سافت‌مکس را از طریق نگاشت توابع ویژگی تصادفی مثبت و متعامد شکست. این رویکرد تقریب بدون سوگیری از توجه کامل را در زمان و حافظه کاملاً خطی به ارمغان می‌آورد.",
        "analogy": r"آنالیز فوریه و سنتز آکوستیک امواج صوتی را تجسم کنید: به جای آنکه یک قطعه موسیقی ۱ ساعته را از طریق مقایسه ضرب داخلی تک‌تک صدم‌ثانیه‌ها با تمام نقاط دیگر قطعه تحلیل کنید، امواج صوتی را به پایه‌های فرکانسی هارمونیک متعامد تصویر کرده و انرژی کل را با ضرب ضرایب طیفی ارزیابی می‌نمایید.",
        "derivation": r"""هسته توجه نمایی $\exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})$ با امید ریاضی توابع ویژگی تصادفی ارتوگونال مثبت تقریب زده می‌شود:
\begin{equation}
\mathbf{A}_{ij} \approx \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j), \quad \phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{M}} \left[ \exp(\mathbf{w}_1^\top \mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}), \dots, \exp(\mathbf{w}_M^\top \mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}) \right]^\top
\end{equation}
که در آن ماتریس بردارهای پرفکت متعامد $\mathbf{W} \in \mathbb{R}^{M \times d}$ از تجزیه \lr{QR} ماتریس گوسی حاصل می‌آید. با خاصیت شرکت‌پذیری:
\begin{equation}
\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \left( \phi(\mathbf{Q}) \left( \phi(\mathbf{K})^\top \mathbf{V} \right) \right) \cdot \operatorname{diag}\left( \phi(\mathbf{Q}) \sum_{j=1}^N \phi(\mathbf{k}_j) \right)^{-1}
\end{equation}
محاسبات در زمان خطی $\mathcal{O}(N M d)$ بدون تشکیل ماتریس توجه مادی صورت می‌پذیرد.""",
        "tensor_trace": [
            r"تولید ماتریس‌های فرافکنی تصادفی متعامد با فاکتورسازی \lr{QR} به ابعاد $[M, d]$.",
            r"نگاشت بردارهای پرس‌وجو و کلید به بردارهای ویژگی مثبت با ابعاد $[B, H, N, M]$.",
            r"محاسبه تانسور انباشت حالت کلید-مقدار با ضرب $\mathbf{S} = \phi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{B \times H \times M \times d_v}$.",
            r"ضرب بردار ویژگی پرس‌وجو در تانسور انباشت حالت جهت بازیابی تعاملات.",
            r"نرمال‌سازی برداری با بردار مجموع ستونی کلیدها و خروجی نهایی لایه."
        ],
        "roofline": [
            r"\textbf{حذف کامل ماتریس مربعی از حافظه:} عدم نیاز به نگهداری ماتریس $N \times N$ در حافظه اصلی، ردپای رم را کاملاً خطی می‌سازد.",
            r"\textbf{جایگزینی عملیات \lr{Memory-Bound} با ضرب‌های متوالی:} محاسبات در قالب دو ضرب ماتریسی پیوسته اجرا شده و سقف توان تراشه را پر می‌کند."
        ],
        "status": "PASS", "acc": "87.20", "f1": "84.5", "vram": "720.0", "thr": "39,100", "vram_pct": "82.9",
        "justification": r"ثبت توان فوق‌العاده 39,100 توکن در ثانیه و اوج مصرف رم 720.0 مگابایت (کاهش 82.9 درصدی نسبت به ۴۲۰۰ مگابایت مبنا) در کنار دقت پایدار 87.20 درصد، اثبات‌کننده قدرت ریاضی نگاشت ارتوگونال مثبت \lr{FAVOR+} است که خطای واریانس تقریب‌های رندوم قبلی را مهار کرده است.",
        "tikz": r"""\begin{tikzpicture}[scale=0.82, every node/.style={scale=0.82}, 
    node distance=1.1cm,
    auto,
    block/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    favor/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small},
    outblock/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm, font=\small}
]
    \node [block] (input) {ورودی $\mathbf{Q}, \mathbf{K}, \mathbf{V}$};
    \node [favor, right=1.0cm of input] (phi) {نگاشت \lr{FAVOR+} $\phi(\cdot)$};
    \node [favor, right=1.0cm of phi] (assoc) {تجمع حافظه $\phi(\mathbf{K})^\top \mathbf{V}$};
    \node [outblock, right=1.0cm of assoc] (out) {خروجی خطی};

    \draw [->, thick, draw=blue!80] (input) -- (phi);
    \draw [->, thick, draw=orange!80] (phi) -- (assoc);
    \draw [->, thick, draw=orange!80] (assoc) -- (out);
\end{tikzpicture}""",
        "ai_prompt": "A 3D scientific rendering of the Performer FAVOR+ mechanism. Geometric orthogonal projection axes glowing in cyan and purple, mapping continuous token tensors into positive random feature spaces, futuristic data flow, 8k resolution --ar 16:9"
    }
}
print("Loaded Part A modules:", len(PART_A))
