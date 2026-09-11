import os
import csv

results_file = '/Users/tahamajs/Documents/uni/Project/top_100_results.csv'
names = []
if os.path.exists(results_file):
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 10:
                names.append(row[0])

# Fallback names if file reading fails
if not names:
    names = ['nli_softmax', 'weighted_ensemble', 'frandi', 'slicegpt']

def get_structural_description(name):
    name_lower = name.lower()
    
    # 1. Softmax & Linear Attention
    if 'softmax' in name_lower or 'nasc' in name_lower or 'linear' in name_lower and 'hashed' not in name_lower and 'cur' not in name_lower:
        return r"""ساختار این مکانیزم توجه بر پایه‌ی تقریب توابع غیرخطی بنا شده است. در مکانیزم استاندارد توجه، محاسبه‌ی ماتریس احتمال نیازمند اعمال تابع نمایی بر روی ضرب داخلی $\mathbf{Q} \mathbf{K}^T$ است که دارای پیچیدگی $\mathcal{O}(N^2)$ می‌باشد. این ماژول با بهره‌گیری از نگاشت‌های ویژگی (Feature Maps) نظیر $\phi(\cdot)$، تقریب $\exp(\mathbf{q}_i^T \mathbf{k}_j) \approx \phi(\mathbf{q}_i)^T \phi(\mathbf{k}_j)$ را اعمال می‌کند. با استفاده از خاصیت شرکت‌پذیری ضرب ماتریسی، محاسبات به فرم $\phi(\mathbf{Q}) (\phi(\mathbf{K})^T \mathbf{V})$ بازآرایی می‌شود که منجر به کاهش پیچیدگی زمانی و مکانی به فرم خطی $\mathcal{O}(N \cdot d^2)$ می‌گردد. همچنین برای حفظ پایداری عددی، از روش‌های نرمال‌سازی متقارن در مخرج کسر توجه استفاده می‌شود."""
    
    # 2. Ensembles
    elif 'ensemble' in name_lower or 'smart' in name_lower:
        return r"""معماری این مکانیزم از ترکیب چندین زیرشبکه یا هدِ توجه (Attention Branches) مجزا تشکیل شده است. به جای تکیه بر یک ماتریس توجه واحد، ورودی $\mathbf{X}$ به چندین مسیر موازی هدایت می‌شود که هر یک ویژگی‌های محلی یا سراسری متفاوتی را استخراج می‌کنند. خروجی نهایی به صورت یک ترکیب خطی وزن‌دار $\mathbf{O} = \sum_{i=1}^{M} \alpha_i \mathrm{Attention}_i(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ محاسبه می‌گردد. مقادیر دروازه‌بندی (Gating) $\alpha_i$ می‌توانند به صورت پارامترهای آموزش‌پذیر، تابعی از ورودی، یا مکانیزم‌های مسیریابی (Routing) مبتنی بر شبکه عصبیِ کم‌عمق تعیین شوند. این ساختار تضمین می‌کند که در صورت افت کیفیت در یک زیرشبکه، سایر زیرشبکه‌ها با وزن‌دهی تطبیقی جبران خطا کنند."""
    
    # 3. Random Projections
    elif 'frandi' in name_lower or 'rand' in name_lower:
        return r"""در قلب این معماری، تئوری پرتاب تصادفی جانسون-لیندن‌اشتراوس (Johnson-Lindenstrauss Lemma) پیاده‌سازی شده است. برای کاهش ابعاد ماتریس‌های پرس‌وجو ($\mathbf{Q}$) و کلید ($\mathbf{K}$)، از ماتریس‌های تصادفی متعامد $\mathbf{R} \in \mathbb{R}^{d \times k}$ استفاده می‌شود که در آن $k \ll d$. با اعمال این پرتاب تصادفی ($\hat{\mathbf{Q}} = \mathbf{Q} \mathbf{R}$)، فاصله کسینوسی و ضرب داخلی بردارهای توجه در فضای کم‌بعدتر تا حد بسیار بالایی حفظ می‌شود. این فشرده‌سازی باعث می‌شود تا عملیات محاسبه شباهت با سرعت بالاتر و حافظه کمتری انجام شود، ضمن آنکه ویژگی‌های هندسی فضای تعبیه‌سازی اولیه دستخوش تغییرات مخرب نگردد."""
    
    # 4. Pruning / Sparsity
    elif 'prun' in name_lower or 'slice' in name_lower or 'armor' in name_lower or 'cure' in name_lower or 'come' in name_lower:
        return r"""این ماژول از تکنیک‌های هرس ساختاریافته (Structured Pruning) برای خلوت‌سازی (Sparsification) ماتریس‌های وزن $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$ استفاده می‌کند. با ارزیابی میزان اهمیت هر سطر یا ستون (اغلب از طریق بسط تیلورِ تابع زیان یا محاسبه نرم مقادیر ویژگی)، ابعاد زائد از ماتریس‌ها به طور کامل حذف (Slice) می‌شوند. برخلاف هرس‌های غیرساختاریافته که نیازمند فرمت‌های پیچیده‌ی حافظه هستند، این روش ابعاد پنهان (Hidden Dimension) مدل را مستقیماً کاهش داده و به ترانسفورمر اجازه می‌دهد تا ضرب‌های ماتریسی متراکم (Dense GEMM) را با ماتریس‌های کوچکتر انجام دهد، که منجر به کاهش خطی در تعداد پارامترهای محاسباتی (FLOPs) می‌شود."""
    
    # 5. Quantization & Hashing
    elif 'hash' in name_lower or 'flat' in name_lower or 'roast' in name_lower or 'int4' in name_lower or 'uniql' in name_lower or 'fp8' in name_lower:
        return r"""ساختار این ماژول بر پایه فشرده‌سازی مقادیر عددی از طریق درهم‌سازی (Hashing) و کوانتیزاسیون (Quantization) استوار است. در مکانیزم توجه، ماتریس‌های وزن به جای ذخیره‌سازی با دقت ۳۲ بیتی، به کلاسترهای مجزا تقسیم شده و مقادیر آن‌ها در قالب‌های کم‌دقت‌تر (مانند \lr{INT4} یا \lr{FP8}) ذخیره می‌شوند. برای جبران خطای ناشی از این فشرده‌سازی، از تکنیک‌های چرخش فضا (Rotational Matrices) یا کالیبراسیون توزیع مقادیر (Outlier Handling) استفاده می‌گردد. همچنین در برخی از این معماری‌ها، از توابع درهم‌ساز برای به اشتراک‌گذاری وزن‌ها (Weight Sharing) میان هد‌های مختلف توجه استفاده می‌شود که افزونگی حافظه را به شدت می‌کاهد."""
    
    # 6. Low-Rank / SVD / TT
    elif 'svd' in name_lower or 'soka' in name_lower or 'subspace' in name_lower or 'cur' in name_lower or 'kronecker' in name_lower or 'tt' in name_lower or 'lora' in name_lower:
        return r"""معماری این ساختار بر پایه تقریب‌های رتبه-پایین (Low-Rank Approximations) نظیر تجزیه مقادیر منفرد ($\mathbf{SVD}$) یا تجزیه کرونکر (Kronecker) استوار است. ماتریس پرپارامتر وزن $\mathbf{W}$ از طریق فاکتورسازی به ضرب دو یا چند ماتریس کوچکتر $\mathbf{W} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ تقریب زده می‌شود. با حفظ تنها $r$ مقدار منفرد برتر (جایی که $r \ll d$)، ظرفیت بازنمایی شبکه حفظ شده اما تعداد پارامترها به شدت کاهش می‌یابد. در حین استنتاج، به جای ضرب مستقیم ورودی در $\mathbf{W}$، ورودی ابتدا به یک زیرفضای کم‌بعد (از طریق $\mathbf{V}^T$) پرتاب شده و پس از پردازش، توسط $\mathbf{U}$ به فضای اصلی بازگردانده می‌شود."""
    
    # 7. Encoding / Compression (Huffman, GreCoDeC)
    elif 'codec' in name_lower or 'huffman' in name_lower or 'enlighten' in name_lower:
        return r"""در این مدل، ساختار توجه با یک لایه کدگذاری انتروپی (Entropy Coding) ترکیب شده است. پس از محاسبه ماتریس توجه، توزیع مقادیر احتمال و اکتویشن‌ها تحلیل شده و از کدهایی با طول متغیر (مانند کدگذاری هافمن) برای تخصیص بیت‌های کمتر به مقادیر پرتکرار (نظیر وزن‌های صفر در ماتریس‌های تنک) استفاده می‌شود. این فشرده‌سازی بدون اتلاف (Lossless) یا با اتلاف کنترل‌شده، پهنای باند حافظه (Memory Bandwidth) مورد نیاز برای بارگذاری و واکشی تنسورها از حافظه‌ی اصلی (DRAM) به حافظه‌ی نهان (SRAM) را به طرز چشمگیری کاهش داده و گلوگاه ورودی/خروجی (I/O Bottleneck) را رفع می‌کند."""
    
    # 8. Recurrent / SSM
    elif 'mamba' in name_lower or 'ssm' in name_lower or 'hgrn' in name_lower or 'delta' in name_lower:
        return r"""این ماژول، مکانیزم توجه را به عنوان یک سیستم پویای بازگشتی (Linear Recurrent System) و مدل‌های فضای حالت فرموله می‌کند. برخلاف توجه مربعی، این ساختار حالت پنهانی $\mathbf{h}_t$ را نگهداری می‌کند که با هر توکن ورودی به صورت $\mathbf{h}_t = \mathbf{A} \mathbf{h}_{t-1} + \mathbf{B} \mathbf{x}_t$ بروزرسانی می‌شود. با اعمال ماتریس‌های انتقال وابسته به داده (Data-Dependent Gating) و استفاده از عملگرهای اسکن تجمعی در سطح سخت‌افزار (Associative Scan)، این معماری به حافظه ثابت $\mathcal{O}(1)$ در استنتاج دست یافته و توانایی ردیابی وابستگی‌های طولانی‌مدت (Long-term Dependencies) را بدون نیاز به ذخیره کامل \lr{KV-Cache} فراهم می‌آورد."""
    
    # 9. Hardware Fused Attention
    elif 'soa' in name_lower or 'fused' in name_lower or 'flash' in name_lower:
        return r"""ساختار این عملگر متمرکز بر بهینه‌سازی دسترسی به حافظه از طریق همجوشی کرنل‌ها (Kernel Fusion) است. به جای اجرای مرحله به مرحله ضرب ماتریس‌ها، نرمال‌سازی Softmax و ضرب نهایی که همگی نیازمند خواندن و نوشتن‌های مکرر در حافظه میانی (HBM) هستند، این معماری محاسبات را در بلاک‌های کوچک (Tiling) دسته‌بندی کرده و کل زنجیره محاسباتی را مستقیماً درون حافظه بسیار سریع SRAM روی پردازنده گرافیکی (GPU) اجرا می‌کند. این تغییر ساختاری از نظر ریاضی خروجی دقیقاً یکسانی با توجه استاندارد تولید می‌کند، اما به دلیل افزایش نمایی در شدت محاسباتی (Arithmetic Intensity)، مشکل دیواره‌ی حافظه را دور می‌زند."""
    
    # Fallback Generic Attention variant
    else:
        return r"""ساختار این ماژول، تعمیمی بر مکانیزم توجه نقطه‌ای (Dot-Product Attention) با تغییر در نحوه مسیریابی و نرمال‌سازی تنسورها می‌باشد. ماتریس‌های $\mathbf{Q, K, V}$ با عبور از نگاشت‌های تحویلی، در فضایی با پیچیدگی پایین‌تر ادغام می‌شوند. بازآرایی محاسبات و استفاده از ماسک‌های پویای علی (Dynamic Causal Masking) در این معماری تضمین می‌کند که جریان گرادیان در حین بازپس‌گیری دچار محوشدگی (Vanishing Gradient) نگردد و ساختار ماتریس توجهِ حاصل، تعادل مطلوبی میان ظرفیت یادگیری و هزینه سخت‌افزاری ایجاد کند."""


tex_code = r"""\documentclass[12pt,a4paper]{report}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{geometry}
\geometry{a4paper, left=2.5cm, right=3cm, top=3cm, bottom=3cm}
\usepackage{xepersian}
\settextfont{X Nazanin}
\setdigitfont{X Nazanin}
\setlatintextfont{Times New Roman}

\begin{document}

\chapter*{تشریح ساختار معماری مدل‌های توجه و فشرده‌سازی}
\addcontentsline{toc}{chapter}{تشریح ساختار معماری مدل‌های توجه و فشرده‌سازی}

در این بخش، ساختار معماری، فرمول‌بندی ریاضی و مکانیزم‌های عملکردی ماژول‌های بررسی شده به تفصیل و تک‌به‌تک (به تفکیک) تشریح می‌گردند. در این توضیحات، تمرکز مستقیماً بر روی طراحی لایه‌ها، نحوه مدیریت ماتریس‌های پرس‌وجو، کلید و مقدار (\lr{Q, K, V})، و نحوه تقلیل پیچیدگی مکانی و زمانی معطوف است.

"""

for i, name in enumerate(names):
    # Escape underscores for LaTeX text
    safe_name = name.replace('_', '\\_')
    desc = get_structural_description(name)
    tex_code += f"\\section*{{{i+1}. ساختار ماژول \\lr{{{safe_name}}}}}\n"
    tex_code += f"\\addcontentsline{{toc}}{{section}}{{{i+1}. ساختار ماژول \\lr{{{safe_name}}}}}\n"
    tex_code += desc + "\n\n"

tex_code += r"\end{document}"

with open('/Users/tahamajs/Documents/uni/Project/attention_structures.tex', 'w', encoding='utf-8') as f:
    f.write(tex_code)

print("Generated attention_structures.tex successfully.")
