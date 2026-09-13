import os

latex_content = r"""\documentclass[12pt,a4paper]{report}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{a4paper, left=2.5cm, right=3cm, top=3cm, bottom=3cm}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{setspace}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{color, colortbl}
\usepackage{float}
\usepackage{xepersian}

\settextfont{X Nazanin}
\setdigitfont{X Nazanin}
\setlatintextfont{Times New Roman}

\makeatletter
\def\UseMathForPositioningText{}
\def\@kernel@tabular@init{}
\makeatother

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=cyan,
    pdftitle={NeuralOps: A Comprehensive Treatise on Attention Mechanisms}
}

\begin{document}

\begin{titlepage}
    \centering
    \includegraphics[height=3.2cm]{logo.png} \hspace{2cm}
    \includegraphics[height=3.2cm]{logo-en.png}
    
    \vspace{1cm}
    {\large\bfseries دانشگاه تهران}\\[0.3cm]
    {\large\bfseries دانشکدگان دانشکده‌های فنی}\\[0.3cm]
    {\large\bfseries دانشکده مهندسی برق و کامپیوتر}\\[1.5cm]
    
    {\Large\bfseries پایان‌نامه کارشناسی مهندسی کامپیوتر}\\[1.2cm]
    
    \rule{\linewidth}{1.2pt}\\[0.5cm]
    {\LARGE\bfseries NeuralOps: A Hardware-Aware Framework for Scalable SVD Linear Attention and Transformer Model Compression}\\[0.5cm]
    \rule{\linewidth}{1.2pt}\\[1.8cm]
    
    \begin{tabular}{rl}
        \textbf{نگارش:} & \Large\textbf{محمد طه مجلسی} \\
        \textbf{شماره دانشجویی:} & \large\textbf{810101504} \\
        \textbf{استاد راهنما:} & \Large\textbf{دکتر احمد کلهر} \\
    \end{tabular}
    
    \vfill
    {\large\bfseries شهریور ۱۴۰۵}
\end{titlepage}

\tableofcontents
\listoffigures
\clearpage

\chapter{متدولوژی آزمایشگاهی، نتایج بنچ‌مارک و انتخاب مدل قهرمان}

\section{هدف پژوهش و رویکرد غربال‌گری ۱۰۰ مکانیزم توجه}
هدف بنیادین در این پروژه پژوهشی، پیاده‌سازی و یکپارچه‌سازی بیش از ۱۰۰ مکانیزم توجه متنوع و فشرده‌سازی در چارچوب یکپارچه \lr{NeuralOps}، اعتبارسنجی قابلیت تعویض بلادرنگ آن‌ها روی مدل‌های ترنسفورمر، و در نهایت گزینش، تحلیل و استقرار برترین معماری به عنوان مدل قهرمان (\lr{Champion Model}) بوده است. تمامی آزمایش‌ها در شرایط آزمایشگاهی سخت‌افزاری کاملاً یکسان روی پردازنده گرافیکی \lr{NVIDIA RTX 4070 (8GB VRAM)} با شتاب‌دهنده کرنل \lr{Triton 3.0} و محیط \lr{PyTorch 2.5} روی مجموعه داده استاندارد \lr{GLUE SST-2} انجام پذیرفته است.

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{combined_hero_architecture.jpg}
\caption{تصویر یکپارچه انتزاعی سخت‌افزار و نمودار جریان داده در معماری قهرمان Fusion Attention}
\end{figure}

\section{معرفی و تشریح عمیق مدل قهرمان پروژه: Fusion Attention}
بر اساس ارزیابی‌های جامع و پردازش‌های تجربی، مکانیزم \lr{Fusion Attention (FUSIONATTN)} به عنوان برترین و پیروزمندترین مدل در میان تمامی مکانیزم‌ها برگزیده شد. این ماژول توانست به دقت فوق‌العاده \textbf{۹۳.۱۲٪} دست یابد که حتی نسبت به مدل مبنای استاندارد (\lr{Baseline} با ۹۱.۳۰٪) رشد ۲ درصدی در صحت بازنمایی نشان می‌دهد؛ در حالی که مصرف حافظه \lr{VRAM} را از ۴۲۰۰ مگابایت به کمتر از ۴۸۰ مگابایت (بیش از ۸۸٪ صرفه‌جویی در حافظه) کاهش داده است.

\subsection{سازوکار ریاضی و معماری Fusion Attention}
در این معماری، ضرب سنتی توجه با یک سیستم گیت‌بندی بلوکی انطباقی (\lr{Block-Level Gating}) و تقریب زیرفضای SVD ادغام شده است. ماتریس شباهت بلوکی به فرم زیر ارزیابی می‌گردد:
\[
\mathbf{S}_{\text{blk}} = \frac{\mathbf{G}_Q(\mathbf{Q}_{\text{blk}}) \mathbf{G}_K(\mathbf{K}_{\text{blk}})^T}{\tau}
\]
که در آن $\tau$ دمای یادگیرنده گیت و $\mathbf{G}_Q, \mathbf{G}_K$ شبکه‌های تصویرسازی تنش‌های محلی هستند. بلوک صفر که شامل توکن نشانگر \lr{[CLS]} است، همواره در محاسبات باقی می‌ماند و سایر بلوک‌ها با مکانیزم \lr{Top-k} هوشمند انتخاب می‌گردند. این فرآیند مانع از محاسبات برهم‌کنش جفتی غیرضروری میان توکن‌های کم‌اهمیت گشته و اجرای عملیات را مستقیماً در کش سریع \lr{SRAM} میسر می‌سازد.

\section{تحلیل نموداری مسابقات فینال و مقایسه با Baseline}
در شکل‌های زیر، برترین مدل‌های استخراج‌شده در فینال مقایسه‌ای در برابر خط مبنای استاندارد نمایش داده شده‌اند:

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{chart_champion_comparison.png}
\caption{رتبه‌بندی نهایی ۱۰ مکانیزم برتر در مقایسه با خط آستانه مدل مبنا (Baseline 91.30\%)}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{chart_pareto_vram.png}
\caption{نمودار مصالحه دقت و حافظه (Pareto Trade-off) نشان‌دهنده موقعیت استثنایی Fusion Attention}
\end{figure}

\clearpage
\chapter{تشریح جامع و تک‌به‌تک مکانیزم‌های توجه بر حسب رده‌های معماری}
در این فصل، تمامی ۱۳۶ مکانیزم توجه بر اساس تئوری محاسباتی در ۵ رده بنیادین به همراه تشریح کامل مکانیزم، فرمول‌بندی، کارنامه تجربی و توجیه علمی نتایج ارزیابی می‌گردند.

\input{attentions_latex/part1.tex}
\clearpage
\input{attentions_latex/part2.tex}
\clearpage
\input{attentions_latex/part3.tex}
\clearpage
\input{attentions_latex/part4.tex}
\clearpage
\input{attentions_latex/part5.tex}

\chapter{جمع‌بندی و چشم‌انداز آینده}
در این پژوهش جامع، بیش از ۱۰۰ مکانیزم متنوع ارزیابی شدند و نشان داده شد که معماری \lr{Fusion Attention} با ادغام گیت‌بندی بلوکی و همجوشی کرنل‌های ترایتون، مناسب‌ترین و قدرتمندترین جایگزین برای مکانیزم توجه درجه دو استاندارد در پردازش مدرن زبان طبیعی است. با مهار حافظه تا سقف ۸۸٪، افق‌های نوینی برای استقرار مدل‌های زبانی حجیم روی دستگاه‌های لبه شبکه گشوده شد.

\clearpage
\appendix
\chapter{پیوست‌ها: راهنمای ساخت تصاویر با پرامپت‌های مهندسی‌شده}
\section{پرامپت‌های استاندارد برای تولید گرافیک‌های علمی هوش مصنوعی}
برای تصویرسازی تخصصی و الحاق گرافیک‌های نهایی به جایگاه‌های تعبیه‌شده در فصل دوم، پرامپت‌های مهندسی‌شده زیر برای تولید تصویر در مدل‌های هوش مصنوعی تدوین شده‌اند:

\subsection*{۱. پرامپت رده توجه اسپکترال و فاکتورسازی SVD}
\begin{latin}
\small
A high-tech scientific diagram illustrating Singular Value Decomposition (SVD) applied to neural network attention weight matrices. A large dense matrix splits into three matrices: Orthogonal Left Singular Vectors U, Diagonal Singular Value Matrix Sigma, and Right Singular Vectors V-transpose. Glowing cyan energy truncation showing discarded small singular values. Dark academic background, crisp lines, modern AI hardware architecture style.
\end{latin}

\subsection*{۲. پرامپت رده توجه خطی و نگاشت‌های کرنلی Softmax}
\begin{latin}
\small
An educational academic diagram of Linear Kernel Attention. Shows the mathematical associative property transformation: swapping (Q x K transpose) x V with quadratic O(N squared) complexity into Q x (K transpose x V) with linear O(N) complexity using non-linear kernel feature maps phi(x). Neon blue and amber matrix visualization with flow arrows on dark background.
\end{latin}

\subsection*{۳. پرامپت رده مدل‌های فضای حالت (SSM) و شبکه‌های بازگشتی}
\begin{latin}
\small
Schematic of a modern Selective State Space Model (SSM) and Gated Linear RNN. Shows token inputs feeding into a hidden continuous state h t updating sequentially with data-dependent decay gates A and B, eliminating KV cache with constant O(1) inference memory. Vibrant emerald green and cyan signals running through digital registers on dark background.
\end{latin}

\subsection*{۴. پرامپت رده تنک‌سازی و درهم‌سازی حساس به مکان (LSH)}
\begin{latin}
\small
A high-contrast visual diagram of Sparse and Locality Sensitive Hashing (LSH) Attention. A 2D attention matrix divided into colored clusters and sparse diagonal block patterns, showing tokens bucketed by angular hyperplanes and attending only to relevant token subsets. Purple, orange and deep blue hues, clean technical design.
\end{latin}

\subsection*{۵. پرامپت رده همجوشی سخت‌افزاری در SRAM و بهینه‌سازی حافظه}
\begin{latin}
\small
Hardware architecture schematic showing GPU chip memory hierarchy: High-Bandwidth Memory (HBM/DRAM) transferring tiled blocks into ultra-fast on-chip SRAM cache. Inside SRAM, FlashAttention and Triton fused kernels execute matrix multiplication and online softmax without intermediate memory round-trips. Glowing copper circuits, cyan data pipelines.
\end{latin}

\section{پروتکل خط لوله بنچ‌مارک خودکار و بازآفرینی نتایج}
کلیه نتایج گزارش‌شده در این رساله از طریق فرمان‌های زیر قابل بازآفرینی و سنجش مجدد می‌باشند:
\begin{latin}
\begin{verbatim}
# Run champion evaluation on GPU
python Codes/framework/eval_all_models_datasets.py \
    --model bert-base-uncased \
    --dataset sst2 \
    --attn_type fusion_attn \
    --rank 32 \
    --device cuda
\end{verbatim}
\end{latin}

\end{document}
"""

with open('/Users/tahamajs/Documents/uni/Project/thesis_final/master_thesis.tex', 'w', encoding='utf-8') as f:
    f.write(latex_content)

print("master_thesis.tex regenerated successfully.")
