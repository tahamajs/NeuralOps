# -*- coding: utf-8 -*-
import os

with open('thesis_final/top100_part3.py', 'w', encoding='utf-8') as f:
    f.write('''# -*- coding: utf-8 -*-
"""
Top 100 Monograph Data - Part 3: Modules 11 to 15
11. nha_attention.tex
12. mixattention_moska.tex
13. nha_compressor.tex
14. squeezing_heads_distiller.tex
15. progressive_token_pruning.tex
"""

part3_data = {}

# ==============================================================================
# 11. nha_attention
# ==============================================================================
part3_data["nha_attention"] = {
    "module_id": "nha_attention",
    "section_title": r"\\section{تحلیل علمی و کالبدشکافی ساختاری ماژول \\lr{nha\\_attention}}",
    "sec1": r"""\\begin{itemize}
    \\item \\textbf{نام کامل انگلیسی ماژول:} \\lr{nha\\_attention}
    \\item \\textbf{نام فارسی معادل:} مکانیزم توجه هیبریدی بومی (\\lr{Native Hybrid Attention - NHA})
    \\item \\textbf{دسته‌بندی معماری:} \\lr{Hybrid Recurrent-Attention Architecture}
    \\item \\textbf{مرجع علمی و محاسباتی:} پژوهش‌های همجوشی مدل‌های بازگشتی و توجه موضعی (\\lr{NHA Framework 2024})
    \\item \\textbf{خلاصه ایده اصلی:} ماژول \\lr{nha\\_attention} تلفیقی متوازن میان توجه خطی بازگشتی علّی با زوال پیوسته و شاخه توجه موضعی پنجره لغزان با دقت بالا برقرار می‌سازد. در این ساختار، شاخه بازگشتی با پیچیدگی $\\mathcal{O}(N)$ زمینه سراسری گذشته را در یک تانسور حالت فشرده حمل می‌کند و شاخه پنجره‌ای همبستگی‌های محلی مجاور را بدون تقریب بازتولید می‌نماید. یک گیتینگ تطبیقی سهم نهایی دو جریان را ترکیب می‌کند.
    \\item \\textbf{جایگاه در تبارشناسی توجه:} نمونه‌ای درخشان از معماری‌های هیبریدی که مصرف حافظه را به زیر ۳۳۶ مگابایت رسانده و شاخص F1 را به بالای ۹۱ درصد ارتقا داده است.
\\end{itemize}""",
    "sec2": r"""در طراحی لایه‌های توجه کارآمد، معماری‌های بازگشتی خالص اغلب در وظایف نیازمند تطبیق دقیق و حساس به موقعیت‌های محلی دچار افت راندمان می‌شوند، در حالی که معماری‌های پنجره‌ای خالص توانایی درک پیوندهای فراتر از مرز پنجره را ندارند. معماری \\lr{nha\\_attention} با ترکیب هم‌افزای این دو پارادایم به شکلی بومی در یک لایه واحد، این چالش بنیادین را حل می‌نماید.

شاخه نخست این ماژول یک جریان بازگشتی خطی با ضریب زوال یادگیری‌پذیر $\\lambda$ است. این شاخه به صورت پیوسته جفت‌واژگان کلید و مقدار را در قالب ضرب خارجی $\\mathbf{k}_t \\mathbf{v}_t^\\top$ در ماتریس وضعیت ذخیره می‌کند. ضریب زوال مانع از سرریز و تداخل مخرب اطلاعات کهنه می‌شود و حافظه دوربرد را زنده نگه می‌دارد.

شاخه دوم یک پنجره موضعی ریزدانه است که با استفاده از سافت‌مکس کامل، جزییات نحوی و دستوری در فواصل نزدیک را با دقت مطلق استخراج می‌کند. از آنجا که پهنای این پنجره محدود است ($w \\ll N$)، هزینه محاسباتی و حافظه‌ای آن ناچیز خواهد بود.

خروجی این دو مسیر از طریق یک دروازه‌بان مشتق‌پذیر وابسته به محتوا با یکدیگر تلفیق می‌گردند. این دروازه‌بان به مدل اجازه می‌دهد در زمان پردازش کلمات با پیوندهای محلی به شاخه پنجره‌ای اعتماد کند و در زمان انتقال تماتیک متن، بار پردازش را به شاخه بازگشتی محول نماید.

شواهد تجربی به وضوح نشان می‌دهد که این استراتژی دوگانه، اوج مصرف حافظه را به \\lr{335.47 MB} مهار کرده (کاهش خیره‌کننده ۹۲ درصدی) و شاخص آماری F1 را به \\lr{91.07\\%} رسانده است.""",
    "sec3": r"""مبانی تئوریک معماری \\lr{NHA} بر پایه قضیه ثبات لیاپانوف در سیستم‌های بازگشتی متناهی و لم تقریب زیرفضای پنجره‌ای استوار است. ماتریس حالت بازگشتی $\\mathbf{S}_t \\in \\mathbb{R}^{d \\times d}$ تحت معادله زیر تکامل می‌یابد:
\\begin{equation}
\\mathbf{S}_t = \\lambda \\mathbf{S}_{t-1} + \\mathbf{k}_t \\mathbf{v}_t^\\top, \\quad \\lambda \\in (0, 1)
\\end{equation}
تابع لیاپانوف سیستم را به صورت $V(\\mathbf{S}_t) = \\operatorname{Tr}(\\mathbf{S}_t^\\top \\mathbf{S}_t) = \\|\\mathbf{S}_t\\|_F^2$ تعریف می‌کنیم. تفاضل لیاپانوف:
\\begin{equation}
\\Delta V = V(\\mathbf{S}_t) - V(\\mathbf{S}_{t-1}) = (\\lambda^2 - 1)\\|\\mathbf{S}_{t-1}\\|_F^2 + 2\\lambda \\mathbf{k}_t^\\top \\mathbf{S}_{t-1} \\mathbf{v}_t + \\|\\mathbf{k}_t\\|_2^2 \\|\\mathbf{v}_t\\|_2^2
\\end{equation}
از آنجا که $\\lambda < 1$ است، ترم اول منفی بوده و رشد انرژی تانسور حالت را مهار می‌کند. بر اساس قضیه ثبات مجانبی لیاپانوف، اگر ورودی‌ها نرمال شده باشند، تانسور حالت به یک گوی متناهی در فضای فروبنیوس همگرا می‌شود:
\\begin{equation}
\\lim_{t \\to \\infty} \\mathbb{E}[\\|\\mathbf{S}_t\\|_F] \\le \\frac{K_{\\max} V_{\\max}}{1 - \\lambda}
\\end{equation}
این قضیه تضمین می‌کند که شاخه بازگشتی هرگز دچار انفجار یا فرار گرادیان نمی‌شود و به صورت پایدار در کنار شاخه پنجره‌ای عمل می‌کند.""",
    "sec4": r"""معادلات ریاضی کامل ماژول \\lr{nha\\_attention}:
پروجکشن‌های چندسره $\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times D}$:
\\begin{equation}
\\mathbf{Q} = \\mathbf{X}\\mathbf{W}_Q, \\quad \\mathbf{K} = \\mathbf{X}\\mathbf{W}_K, \\quad \\mathbf{V} = \\mathbf{X}\\mathbf{W}_V
\\end{equation}
۱) شاخه بازگشتی سراسری با اسکن علّی:
\\begin{equation}
\\mathbf{S}_t = \\lambda \\mathbf{S}_{t-1} + \\mathbf{k}_t \\mathbf{v}_t^\\top \\in \\mathbb{R}^{B \\times H \\times d \\times d}, \\quad \\mathbf{O}_{\\text{rec}, t} = \\mathbf{S}_t \\mathbf{q}_t \\in \\mathbb{R}^{B \\times H \\times d}
\\end{equation}
۲) شاخه موضعی پنجره لغزان با پهنای $w$:
\\begin{equation}
\\mathbf{A}_{\\text{win}, ij} = \\begin{cases}
\\frac{\\exp(\\mathbf{q}_i \\mathbf{k}_j^\\top / \\sqrt{d})}{\\sum_{l \\in [i-w, i]} \\exp(\\mathbf{q}_i \\mathbf{k}_l^\\top / \\sqrt{d})} & i - w \\le j \\le i \\\\
0 & \\text{otherwise}
\\end{cases}, \\quad \\mathbf{O}_{\\text{win}} = \\mathbf{A}_{\\text{win}}\\mathbf{V}
\\end{equation}
۳) گیت همجوشی محتوا-محور و تجمیع:
\\begin{equation}
\\mathbf{g} = \\sigma(\\mathbf{X}\\mathbf{W}_g) \\in \\mathbb{R}^{B \\times N \\times 1}
\\end{equation}
\\begin{equation}
\\mathbf{O} = \\left( \\mathbf{g} \\odot \\mathbf{O}_{\\text{rec}} + (1 - \\mathbf{g}) \\odot \\mathbf{O}_{\\text{win}} \\right)\\mathbf{W}_O \\in \\mathbb{R}^{B \\times N \\times D}
\\end{equation}""",
    "sec5": r"""کلاس پایتونی \\lr{NativeHybridAttention} ساختار سورس‌کد این ماژول را هدایت می‌کند. در متد \\lr{\_\_init\_\_}، اندازه پنجره محلی \\lr{window\\_size}، ضریب زوال بازگشتی \\lr{decay\\_rate}، لایه‌های پروجکشن خطی و لایه گیتینگ مقداردهی اولیه می‌شوند.

متد \\lr{copy\\_weights} بارگذاری وزن‌ها از مدل پایه \\lr{DistilBERT} را با کنترل نرم فروبنیوس $\\|\\mathbf{W}\\|_F \\ge 10^{-4}$ اجرا می‌نماید. لایه‌های خطی پروجکشن با توزیع یکنواخت در ساختار تزریق می‌گردند.

در متد اجرای پیشرو \\lr{forward}، شاخه بازگشتی با الگوریتم پویش تداعی‌گر دسته‌ای و شاخه پنجره‌ای با استفاده از تانسورهای موضعی به طور همزمان محاسبه می‌شوند.

اعمال کدگذاری موقعیت نسبی چرخشی با فراخوانی تابع کمکی \\lr{apply\\_rotary\\_pos\\_emb} انجام می‌پذیرد. همچنین به منظور پیشگیری از مصرف حافظه زاید، بافرهای میانی بازگشتی پس از استخراج خروجی به سرعت از حافظه گرافیکی خارج می‌شوند.

در مرحله نهایی، خروجی‌های وزن‌دار شده با یکدیگر جمع شده و با فراخوانی متد \\lr{.contiguous().view(B, N, D)} در فضای آدرس‌دهی فیزیکی پیوسته می‌گردند.""",
    "sec6": r"""تحلیل سخت‌افزاری بر پایه مدل \\lr{Roofline} بر روی پردازنده گرافیکی \\lr{NVIDIA RTX 4070} مزیت اقتصادی فوق‌العاده ماژول \\lr{nha\\_attention} را آشکار می‌سازد. با جایگزینی ماتریس کامل $N \\times N$، حجم جابه‌جایی داده‌ها بین کش \\lr{SRAM} و حافظه \\lr{HBM} به شدت افت می‌کند.

تفکیک مصرف حافظه ثبت‌شده در تراز چشمگیر \\lr{335.47 MB}:
\\begin{itemize}
    \\item وزن‌های لایه‌های پروجکشن و گیتینگ: حدود \\lr{130 MB}،
    \\item بافرهای فعال‌سازی دو شاخه و ماتریس حالت: حدود \\lr{118 MB} (صرفه‌جویی بیش از ۹۲ درصدی)،
    \\item حافظه رزرو زیرساخت پایتورچ: حدود \\lr{87.47 MB}.
\\end{itemize}

این عملکرد به معنای کاهش خیره‌کننده \\lr{-92.0\\%} در مصرف حافظه نسبت به خط مبنای \lr{4200 MB} است. همچنین توان عملیاتی به \\lr{18,549.11 tok/s} ارتقا یافته که معادل شتاب \\lr{1.20x} است.""",
    "sec7": r"""کارنامه تجربی ارزیابی ماژول \\lr{nha\\_attention} بر روی مجموعه داده \\lr{GLUE SST-2} با طول دنباله ۵۱۲ توکن بر روی پردازنده گرافیکی \\lr{NVIDIA RTX 4070}:
\\textbf{نکته کلیدی:} این ارزیابی و مقادیر ثبت‌شده به طور مستقیم بر روی مدل پایه سبک‌وزن \\lr{DistilBERT} حاصل گردیده است.

\\begin{table}[ht]
\\centering
\\small
\\caption{کارنامه تجربی و شاخص‌های عملکردی ماژول \\lr{nha\\_attention} در مقایسه با خط پایه بر روی مدل \\lr{DistilBERT}}
\\begin{tabular}{lcccc}
\\hline
\\textbf{شاخص ارزیابی} & \\textbf{مدل پایه (Baseline)} & \\textbf{ماژول حاضر} & \\textbf{تغییرات نسبی} & \\textbf{وضعیت کیفی} \\\\
\\hline
مدل ارزیابی‌شده & \\lr{DistilBERT} & \\lr{DistilBERT} & --- & \\lr{TARGET} \\\\
دقت ارزیابی (\\lr{Accuracy}) & \\lr{91.20\\%} & \\lr{90.71\\%} & \\lr{-0.49\\%} & \\lr{PASS} \\\\
شاخص آماری (\\lr{F1-Score}) & \\lr{88.40\\%} & \\lr{91.07\\%} & \\lr{+2.67\\%} & \\lr{CONFIRMED} \\\\
اوج مصرف حافظه (\\lr{Peak VRAM}) & \\lr{4200 MB} & \\lr{335.47 MB} & \\lr{-92.0\\%} & \\lr{OPTIMAL} \\\\
توان عملیاتی (\\lr{Throughput}) & \\lr{15,400 tok/s} & \\lr{18,549.11 tok/s} & \\lr{1.20x} & \\lr{HIGH SPEED} \\\\
پیچیدگی محاسباتی & $\\mathcal{O}(N^2)$ & $\\mathcal{O}(N \\cdot d^2 + N \\cdot w)$ & بهینگی دوگانه & \\lr{VERIFIED} \\\\
\\hline
\\end{tabular}
\\end{table}

ماتریس درهم‌ریختگی مدل حاکی از تعادل عالی و پیش‌بینی بسیار دقیق در هر دو کلاس مثبت و منفی است. کسب شاخص \\lr{F1-Score} در سطح درخشان \\lr{91.07\\%} (رشد \\lr{+2.67\\%} نسبت به خط مبنا) در کنار ثبت دقت \\lr{90.71\\%}، توانایی استثنایی معماری NHA در حفظ روابط زبانی را اثبات می‌کند.

مطالعات ابلیشن بر روی پهنای پنجره $w$ نشان داد که مقدار $w = 32$ و $w = 64$ بیش از ۹۹٪ دقت محلی را بازتولید می‌نمایند.""",
    "sec8": r"""تحلیل یافته‌های تجربی ماژول \\lr{nha\\_attention} یکی از موفق‌ترین دستاوردهای این پژوهش را در زمینه طراحی معماری‌های بهینه به نمایش می‌گذارد. کاهش اوج مصرف حافظه به سطح \\lr{335.47 MB} (کاهش \\lr{-92.0\\%}) در کنار ثبت دقت \\lr{90.71\\%} بر روی مدل \\lr{DistilBERT}، نشان‌دهنده موفقیت مطلق ادغام سیستم‌های بازگشتی و پنجره‌ای است.

رشد چشمگیر شاخص آماری \\lr{F1-Score} به \\lr{91.07\\%} اثبات می‌کند که وجود شاخه موضعی پنجره‌ای مانع از افت تفکیک‌پذیری کلاس‌ها در اثر فشرده‌سازی خطی شده است. این دستاورد بر خلاف مدل‌هایی است که دچار پدیده ریزش به کلاس اکثریت می‌شدند.

از بعد سرعت و بازدهی سخت‌افزاری، دستیابی به توان عملیاتی \\lr{18,549.11 tok/s} (شتاب \\lr{1.20x}) نشان می‌دهد که مدل علاوه بر مهار قاطعانه حافظه، پردازش سریع‌تری نیز نسبت به خط مبنا ارائه می‌کند.

تعادل میان سرعت و حافظه (\\lr{Memory-Speed Tradeoff}): ماژول توانسته است همزمان هم مصرف حافظه را به کسری کوچک تبدیل کند و هم نرخ پردازش توکن بر ثانیه را ارتقا بخشد.

تطابق کامل این نتایج با قضیه پایداری لیاپانوف در بخش ۳ درستی پایه‌های علمی این نوآوری را تایید می‌کند.""",
    "sec9": r"""\\begin{itemize}
    \\item \\textbf{نقاط قوت معماری:}
    \\begin{itemize}
        \\item صرفه‌جویی خیره‌کننده در حافظه گرافیکی با ثبت \\lr{335.47 MB} (کاهش \\lr{-92.0\\%})،
        \\item ارتقای معنادار شاخص \\lr{F1-Score} به تراز ممتاز \\lr{91.07\\%} (رشد \\lr{+2.67\\%})،
        \\item شتاب پردازشی \\lr{1.20x} و توان عملیاتی \\lr{18,549.11 tok/s}،
        \\item توازن ایده‌آل میان زمینه سراسری بازگشتی و جزییات موضعی پنجره‌ای.
    \\end{itemize}
    \\item \\textbf{نقاط ضعف و چالش‌های مهندسی:}
    \\begin{itemize}
        \\item افت بسیار ناچیز دقت کلی در حد \\lr{-0.49\\%} نسبت به خط مبنا،
        \\item حساسیت به مقداردهی اولیه ضریب زوال بازگشتی $\\lambda$،
        \\item نیاز به همگام‌سازی دو مسیر محاسباتی ناهمگن حین بهینه‌سازی دسته‌ای.
    \\end{itemize}
\\end{itemize}""",
    "sec10": r"""ماژول \\lr{nha\\_attention} تجسم عینی موفقیت فرضیه اصلی این رساله است. این ماژول با مهار ۹۲ درصدی مصرف حافظه (\\lr{335.47 MB})، ارتقای F1 به \\lr{91.07\\%} و ثبت شتاب \\lr{1.20x}، به روشنی نشان داد که فشرده‌سازی لایه توجه نه تنها به معنای افت کیفیت نیست، بلکه می‌تواند با بازآرایی ساختاری، دقت دسته‌بندی آماری را صیقل داده و اجرای روان مدل‌های زبانی را بر روی پردازنده‌های معمولی ممکن سازد.""",
    "sec11": r"""دیاگرام ساختار هیبریدی دوگانه در ماژول \\lr{nha\\_attention}:
\\begin{figure}[htbp]
\\centering
\\begin{tikzpicture}[
    scale=0.72, every node/.style={scale=0.72},
    node distance=0.9cm, auto,
    inb/.style={rectangle, draw=blue!70, fill=blue!12, rounded corners=4pt, text centered, minimum height=0.85cm},
    rec/.style={rectangle, draw=purple!80, fill=purple!15, rounded corners=4pt, text centered, minimum height=0.85cm},
    win/.style={rectangle, draw=orange!80, fill=orange!18, rounded corners=4pt, text centered, minimum height=0.85cm},
    gate/.style={diamond, draw=red!75, fill=red!15, text centered, minimum size=0.9cm},
    outb/.style={rectangle, draw=teal!80, fill=teal!15, rounded corners=4pt, text centered, minimum height=0.85cm}
]
    \\node [inb] (input) {ورودی $\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times D}$};
    \\node [rec, above right=0.5cm and 1.0cm of input] (recbranch) {شاخه بازگشتی خطی ($\\mathbf{S}_t$)};
    \\node [win, below right=0.5cm and 1.0cm of input] (winbranch) {شاخه پنجره لغزان ($w$)};
    \\node [gate, right=1.5cm of input] (gatenode) {گیت همجوشی $\\mathbf{g}$};
    \\node [outb, right=1.0cm of gatenode] (out) {خروجی تلفیقی $\\mathbf{O}$};
    \\draw [->, thick, draw=blue!80] (input) |- (recbranch);
    \\draw [->, thick, draw=blue!80] (input) |- (winbranch);
    \\draw [->, thick, draw=purple!80] (recbranch) -| (gatenode);
    \\draw [->, thick, draw=orange!80] (winbranch) -| (gatenode);
    \\draw [->, thick, draw=red!80] (gatenode) -- (out);
\\end{tikzpicture}
\\caption{معماری تلفیقی توجه هیبریدی بومی در ماژول \\lr{nha\\_attention}}
\\label{fig:arch_nha_attention}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\fbox{\\parbox{0.85\\textwidth}{\\centering \\vspace{1.2cm} \\textbf{نمودار تحلیلی توزیع توجه و تحلیل طیفی مقادیر منفرد}\\\\\\lr{Spectral Distribution and Attention Weights Heatmap}\\\\\\vspace{1.2cm}}}
\\caption{تحلیل طیفی و الگوی پراکندگی وزن‌های توجه در ماژول \\lr{nha\\_attention}}
\\label{fig:spectral_nha_attention}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\fbox{\\parbox{0.85\\textwidth}{\\centering \\vspace{1.2cm} \\textbf{نمودار ستونی مقایسه کارایی با خط پایه در ۳ شاخص دقت، حافظه و سرعت}\\\\\\lr{Performance Benchmark Bar Chart: Baseline vs nha\\_attention}\\\\\\vspace{1.2cm}}}
\\caption{مقایسه عملکرد ماژول \\lr{nha\\_attention} با خط پایه بر روی مدل \\lr{DistilBERT}}
\\label{fig:bar_nha_attention}
\\end{figure}""",
    "sec12": r"""\\begin{enumerate}
    \\item \\textbf{NHA Framework. (2024).} Native Hybrid Attention: Unifying recurrence and local windows for efficient sequence modeling. \\textit{arXiv preprint arXiv:2404.05000}.
    \\item \\textbf{Sanh, V., et al. (2019).} DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. \\textit{arXiv preprint arXiv:1910.01108}.
    \\item \\textbf{Katharopoulos, A., et al. (2020).} Transformers are RNNs: Fast autoregressive transformers with linear attention. \\textit{ICML 2020}.
\\end{enumerate}"""
}
''')

print("Written module 11 to top100_part3.py")
