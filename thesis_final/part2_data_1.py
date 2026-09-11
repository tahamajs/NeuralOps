# -*- coding: utf-8 -*-

modules_chunk1 = [
    # 1. flash_attention3_fp8
    r"""
\subsection*{معماری توجه \lr{FlashAttention-3 FP8} با ممیز شناور ۸ بیتی}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{FlashAttention-3 FP8}}

\textbf{مبانی نظری و انگیزه معماری:}
مکانیزم‌های توجه استاندارد چندسره (\lr{Multi-Head Attention}) در ترانسفورمرها که بر پایه دقت‌های ممیز شناور ۱۶ بیتی (\lr{FP16} یا \lr{BF16}) عمل می‌کنند، با چالش جدی مصرف بالای پهنای باند حافظه با پهنای باند بالا (\lr{HBM}) و انباشت ماتریس‌های میانی توجه مواجه هستند. با افزایش طول دنباله ($L$)، تشکیل صریح ماتریس توجه با مرتبه پیچیدگی فضایی $\mathcal{O}(L^2)$ به یک گلوگاه سخت‌افزاری فلج‌کننده تبدیل می‌شود. معماری \lr{FlashAttention-3 FP8} با هدف غلبه بر این محدودیت‌ها بر روی پردازنده‌های گرافیکی نسل هاپر (\lr{NVIDIA Hopper}) طراحی شده است. این معماری الگوریتم‌های کاشی‌کاری دقیق (\lr{Tiling}) را با فرمت کوانتیزاسیون ۸ بیتی ممیز شناور (\lr{FP8}) با ساختار \lr{E4M3} تلفیق می‌کند. با بهره‌گیری از دستورات شتاب‌یافته‌ی سطح سخت‌افزار \lr{WGMMA} (\lr{Warpgroup Matrix Multiply Accumulate})، توان محاسباتی هسته‌های تانسور به دو برابر توان پردازشی \lr{FP16} افزایش یافته و حجم کل تبادل داده میان حافظه مشترک روی تراشه (\lr{SRAM}) و حافظه \lr{HBM} به نصف کاهش می‌یابد.

\textbf{فرمول‌بندی و روابط ریاضی:}
در این معماری، بردار ورودی پنهان $X \in \mathbb{R}^{B \times L \times D}$ ابتدا توسط ماتریس‌های پرتاب به فضاهای پرس‌وجو، کلید و مقدار نگاشت می‌شود. پیش از محاسبه ضرب داخلی، این تنسورها به فرمت ۸ بیتی \lr{E4M3} که شامل ۱ بیت علامت، ۴ بیت توان و ۳ بیت مانتیس است تبدیل می‌گردند:
\begin{equation*}
Q_{\text{fp8}} = \operatorname{cast}_{\text{fp8\_e4m3}}(Q) \in \mathbb{F}_{\text{fp8}}^{B \times L \times H \times d_h}, \quad
K_{\text{fp8}} = \operatorname{cast}_{\text{fp8\_e4m3}}(K), \quad
V_{\text{fp8}} = \operatorname{cast}_{\text{fp8\_e4m3}}(V)
\end{equation*}
محاسبات توجه در قالب بلوک‌های متوالی در حافظه \lr{SRAM} با ابعاد $B_r \times d_h$ برای پرس‌وجو و $B_c \times d_h$ برای کلید-مقدار کاشی‌کاری می‌شوند. برای هر جفت بلوک $(Q_{\text{fp8}, i}, K_{\text{fp8}, j})$، ضرب داخلی مقیاس‌شده به صورت زیر محاسبه می‌گردد:
\begin{equation*}
S_{i, j} = \frac{Q_{\text{fp8}, i} K_{\text{fp8}, j}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B_r \times B_c}
\end{equation*}
به منظور جلوگیری از سرریز عددی و حذف نیاز به ذخیره‌سازی ماتریس کامل در \lr{HBM}، آماره‌های برخط تابع بیشینه نرمال (\lr{Online Softmax}) به صورت مقیاس متغیر درون رجیسترهای \lr{SRAM} بروزرسانی می‌شوند:
\begin{equation*}
m_i^{(j)} = \max\left( m_i^{(j-1)}, \; \operatorname{rowmax}(S_{i, j}) \right)
\end{equation*}
\begin{equation*}
\tilde{P}_{i, j} = \exp\left( S_{i, j} - m_i^{(j)} \right)
\end{equation*}
\begin{equation*}
l_i^{(j)} = \exp\left( m_i^{(j-1)} - m_i^{(j)} \right) \odot l_i^{(j-1)} + \operatorname{rowsum}(\tilde{P}_{i, j})
\end{equation*}
سپس بردار میانی بافت بدون نرمال‌سازی انباشت می‌گردد:
\begin{equation*}
O_i^{(j)} = \operatorname{diag}\left( \exp\left( m_i^{(j-1)} - m_i^{(j)} \right) \right) O_i^{(j-1)} + \tilde{P}_{i, j} V_{\text{fp8}, j}
\end{equation*}
پس از اتمام پیمایش روی کلیه بلوک‌های کلید، نرمال‌سازی نهایی و تبدیل معکوس به دقت مبنا انجام می‌پذیرد:
\begin{equation*}
O_i = \operatorname{cast}_{\text{dtype}}\left( \operatorname{diag}\left( l_i^{(\text{final})} \right)^{-1} O_i^{(\text{final})} \right) \in \mathbb{R}^{B_r \times d_h}
\end{equation*}
\begin{equation*}
Y = O W_{\text{out}} + b_{\text{out}} \in \mathbb{R}^{B \times L \times D}
\end{equation*}

\textbf{سازوکار حافظه و پیچیدگی سخت‌افزاری:}
از دیدگاه پیچیدگی، تعداد کل عملیات‌های ممیز شناور همچنان به صورت $\mathcal{O}(L^2 \cdot d)$ باقی می‌ماند، اما از منظر معماری سامانه، ردپای حافظه \lr{HBM} از مرتبه مربعی به طور کامل حذف شده و تنها به $\mathcal{O}(L \cdot D)$ جهت ذخیره‌سازی بردارهای ورودی و خروجی محدود می‌شود. کلیه ماتریس‌های میانی توجه با ابعاد $B_r \times B_c$ درون حافظه فوق‌سریع \lr{SRAM} مصرف و بازنویسی می‌شوند. استفاده از فرمت \lr{FP8} حجم تبادل داده میان تراشه و حافظه را تا ۵۰ درصد کاهش داده و دستیابی به توان عملیاتی نزدیک به سقف تئوریک پردازنده‌های هاپر را تضمین می‌کند.
""",

    # 2. full_attention
    r"""
\subsection*{معماری توجه استاندارد چندسره (\lr{Full Multi-Head Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه استاندارد چندسره (\lr{Full Attention})}

\textbf{مبانی نظری و انگیزه معماری:}
مکانیزم توجه چندسره بر پایه ضرب داخلی مقیاس‌شده (\lr{Multi-Head Scaled Dot-Product Attention})، سنگ بنای معماری استاندارد ترانسفورمر است که نخستین بار توسط واسوانی و همکاران (۲۰۱۷) معرفی شد. فلسفه اساسی این ساختار، توانمندسازی مدل در برقراری ارتباطات همه‌جانبه میان جفت‌های توکن‌ها در زیرفضاهای بازنمایی گوناگون است. هر هد توجه امکان ردیابی نوع خاصی از روابط نحوی، معنایی، وابستگی‌های متنی کوتاه‌مدت یا بلندمدت را به طور موازی فراهم می‌آورد. با این حال، نیاز ذاتی به محاسبه و ذخیره‌سازی صریح تمامی جفت‌های شباهت، چالش دیواره حافظه و رشد مربعی هزینه‌های پردازشی را برای توالی‌های متنی طولانی به همراه دارد.

\textbf{فرمول‌بندی و روابط ریاضی:}
با فرض بردار ورودی پنهان $X \in \mathbb{R}^{B \times N \times D}$ که در آن $B$ اندازه دسته، $N$ تعداد توکن‌ها و $D$ بعد پنهان مدل است، ابتدا با استفاده از ماتریس‌های آموزش‌پذیر $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$، نگاشت‌های خطی زیرفضا محاسبه می‌شوند:
\begin{equation*}
Q = X W_Q \in \mathbb{R}^{B \times N \times D}, \quad K = X W_K \in \mathbb{R}^{B \times N \times D}, \quad V = X W_V \in \mathbb{R}^{B \times N \times D}
\end{equation*}
سپس این ماتریس‌ها به $H$ هد با بعد $d_k = D / H$ افراز می‌گردند:
\begin{equation*}
Q \in \mathbb{R}^{B \times H \times N \times d_k}, \quad K \in \mathbb{R}^{B \times H \times N \times d_k}, \quad V \in \mathbb{R}^{B \times H \times N \times d_k}
\end{equation*}
امتیازهای خام شباهت از طریق ضرب داخلی مقیاس‌شده میان پرس‌وجوها و کلیدها به دست می‌آید:
\begin{equation*}
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} K_{b, h, j, m} \implies S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times N \times N}
\end{equation*}
در صورت وجود ماسک‌های پدینگ یا ماسک علّی (\lr{Causal Mask})، ماسک افزایشی $M$ اعمال می‌گردد: $\widehat{S} = S + M$. ماتریس احتمالات توجه با اعمال تابع بیشینه نرمال سطری تولید می‌شود:
\begin{equation*}
P = \operatorname{Softmax}(\widehat{S}, \text{dim}=-1) \in \mathbb{R}^{B \times H \times N \times N}, \quad P_{i, j} = \frac{\exp(\widehat{S}_{i, j})}{\sum_{k=1}^N \exp(\widehat{S}_{i, k})}
\end{equation*}
در نهایت، بردارهای زمینه از میانگین‌گیری وزن‌دار مقادیر تشکیل شده و پس از الحاق، از یک لایه پرتاب نهایی عبور می‌کنند:
\begin{equation*}
O = P V \in \mathbb{R}^{B \times H \times N \times d_k}, \quad Y = \operatorname{Concat}(O_1, \dots, O_H) W_O + b_O \in \mathbb{R}^{B \times N \times D}
\end{equation*}

\textbf{سازوکار حافظه و پیچیدگی محاسباتی:}
پیچیدگی محاسباتی زمانی این ماژول $\mathcal{O}(N^2 \cdot d)$ و پیچیدگی فضایی آن $\mathcal{O}(B \cdot H \cdot N^2)$ است. در مرحله آموزش، نگهداری ماتریس فعال‌سازی $P$ برای گام پس‌انتشار گرادیان (\lr{Backward Pass}) نیازمند مصرف صدها مگابایت حافظه \lr{DRAM} برای هر لایه است. در مرحله استنتاج خودهمبسته (\lr{Autoregressive Generation})، کش کلید-مقدار (\lr{KV-Cache}) با نرخ خطی $\mathcal{O}(B \cdot H \cdot N \cdot d_k)$ برای هر توکن جدید رشد می‌کند که برای متون بلند، به ایجاد تنگنای شدید در پهنای باند واکشی حافظه منجر می‌گردد.
""",

    # 3. fused_svd_kernel
    r"""
\subsection*{کرنل همجوش‌یافته خطی \lr{SVD} با نرمال‌سازی مقیم در \lr{SRAM}}
\phantomsection
\addcontentsline{toc}{subsection}{کرنل همجوش‌یافته خطی \lr{SVD} (\lr{Fused SVD Kernel})}

\textbf{مبانی نظری و انگیزه معماری:}
در لایه‌های فشرده‌سازی و پرتاب رتبه-پایین مبتنی بر تجزیه مقادیر منفرد (\lr{SVD})، یک نگاشت ماتریسی پرپارامتر به دو پرتاب متوالی تقلیل می‌یابد: پرتاب رو به پایین به فضای گلوگاه رتبه $r$ ($Z = X V^\top$) و پرتاب بازگشتی رو به بالا ($Y = Z U^\top + b$). در خطوط لوله استاندارد یادگیری عمیق، لایه نرمال‌سازی مانند \lr{RMSNorm} در میانه این دو پرتاب قرار می‌گیرد. این ساختار متوالی موجب بروز سربار سنگین تبادل حافظه (\lr{HBM Roundtrip Overhead}) می‌گردد؛ زیرا تنسور میانی فعال‌سازی $Z \in \mathbb{R}^{M \times r}$ باید به حافظه \lr{DRAM} پردازنده گرافیکی نوشته شده و مجدداً جهت اعمال نرمال‌سازی و پرتاب صعودی واکشی شود. کرنل همجوش‌یافته \lr{Fused SVD-Linear} با بهره‌گیری از محیط برنامه‌نویسی ترایتون (\lr{Triton JIT})، هر سه مرحله پرتاب کاهشی، نرمال‌سازی \lr{RMS} و پرتاب صعودی را در یک کرنل سخت‌افزاری ادغام کرده و تنسور گلوگاهی $Z$ را منحصراً درون رجیسترهای حافظه محلی \lr{SRAM} نگهداری می‌کند.

\textbf{فرمول‌بندی و روابط ریاضی:}
با در نظر گرفتن ماتریس ورودی پهن‌شده $X \in \mathbb{R}^{M \times K_{\text{in}}}$ (که در آن $M = B \cdot N$)، ماتریس پرتاب نزولی $V \in \mathbb{R}^{r \times K_{\text{in}}}$، ماتریس پرتاب صعودی $U \in \mathbb{R}^{K_{\text{out}} \times r}$، بردار مقیاس $\gamma \in \mathbb{R}^r$ و بایاس $b \in \mathbb{R}^{K_{\text{out}}}$، جریان محاسبات مستقیم به صورت زیر پیاده‌سازی می‌گردد:
\begin{enumerate}
    \item \textbf{پرتاب نزولی درون \lr{SRAM}:}
    \begin{equation*}
    Z_{i, :} = \sum_{k=1}^{K_{\text{in}}} X_{i, k} V_{:, k} \in \mathbb{R}^r
    \end{equation*}
    \item \textbf{نرمال‌سازی مجذور میانگین ریشه‌ای درون \lr{SRAM}:}
    \begin{equation*}
    \operatorname{RMS}(Z_i) = \sqrt{\frac{1}{r} \sum_{j=1}^r Z_{i, j}^2 + \epsilon}, \quad \hat{Z}_{i, j} = \frac{Z_{i, j}}{\operatorname{RMS}(Z_i)} \cdot \gamma_j
    \end{equation*}
    \item \textbf{پرتاب صعودی و اعمال بایاس درون \lr{SRAM}:}
    \begin{equation*}
    Y_{i, o} = \sum_{j=1}^r \hat{Z}_{i, j} U_{o, j} + b_o \in \mathbb{R}^{K_{\text{out}}}
    \end{equation*}
\end{enumerate}
در گام پس‌انتشار تحلیلی، گرادیان‌های مربوطه بر اساس مشتقات دقیق بدون ذخیره‌سازی $Z$ در حافظه اصلی به صورت بسته محاسبه می‌شوند:
\begin{equation*}
\frac{\partial \mathcal{L}}{\partial U} = \left(\frac{\partial \mathcal{L}}{\partial Y}\right)^\top \hat{Z}, \quad \frac{\partial \mathcal{L}}{\partial \hat{Z}} = \frac{\partial \mathcal{L}}{\partial Y} U
\end{equation*}
\begin{equation*}
\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{\operatorname{RMS}(Z)} \left( \left(\frac{\partial \mathcal{L}}{\partial \hat{Z}} \odot \gamma\right) - \hat{Z} \odot \frac{1}{r} \sum_{j=1}^r \left(\left(\frac{\partial \mathcal{L}}{\partial \hat{Z}}\right)_j \cdot \gamma_j \cdot \hat{Z}_j\right) \right)
\end{equation*}
\begin{equation*}
\frac{\partial \mathcal{L}}{\partial V} = \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^\top X, \quad \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Z} V
\end{equation*}

\textbf{تحلیل سخت‌افزاری و ارزیابی تجربی:}
این همجوشی کرنلی، نوشتن و خواندن $2 \times M \times r$ داده با دقت ممیز شناور در \lr{HBM} را به صفر می‌رساند. این امر نه تنها سرعت اجرای لایه را تا ۲.۵ برابر نسبت به پیاده‌سازی تفکیک‌شده در پای‌تورچ ارتقا می‌دهد، بلکه با رفع تنگنای پهنای باند حافظه، به راندمان محاسباتی بسیار بالایی دست می‌یابد و برای بهینه‌سازی لایه‌های خطی مدل‌های زبانی حجیم فشرده‌شده با رتبه‌های $r \in [32, 128]$ ایده‌آل است.
""",

    # 4. fusion_attn
    r"""
\subsection*{معماری توجه تنک پیوندی چندپارادایمی (\lr{FUSIONATTN})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{FUSIONATTN}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{FUSIONATTN} یک ماژول توجه ترکیبی و بسیار کارآمد است که بر پایه تجمیع پیش‌بین بلوکی (\lr{Block-Level Predictive Gating}) و توجه تنک انتخابی بنا نهاده شده است. چالش بنیادین در توجه تنک، انتخاب آگاهانه و تفکیک میان توکن‌های حاوی اطلاعات برجسته متنی و توکن‌های کم‌اهمیت بدون تحمیل سربار محاسباتی سنگین است. \lr{FUSIONATTN} توالی ورودی را به بلوک‌های ساختاریافته تقسیم کرده و پیش از محاسبه ماتریس توجه کامل، به کمک یک شبکه گیتینگ سبک، میزان تعامل و همبستگی میان بلوک‌ها را برآورد می‌کند. ویژگی کلیدی این مدل، مهار و تثبیت توکن شاخص کلاس (\lr{[CLS] Token Anchor}) در بلوک صفر است؛ بدین ترتیب همواره جریان بازنمایی سراسری برای وظایف طبقه‌بندی توالی حفظ می‌گردد.

\textbf{فرمول‌بندی و روابط ریاضی:}
با افراز دنباله ورودی به بلوک‌های با طول $B_s$، امتیازهای تقارن بلوکی به صورت زیر ارزیابی می‌شوند:
\begin{equation*}
S_{\text{blk}} = \frac{G_Q(Q_{\text{blk}}) G_K(K_{\text{blk}})^\top}{\tau}
\end{equation*}
که در آن $G_Q$ و $G_K$ توابع فشرده‌ساز خطی بوده و $\tau$ پارامتر دمای کالیبراسیون است. برای هر بلوک پرس‌وجو، تنها $k$ بلوک از برترین کلیدها انتخاب می‌شوند:
\begin{equation*}
\mathcal{I}_{\text{selected}} = \operatorname{TopK}(S_{\text{blk}}, k) \cup \{0\}
\end{equation*}
اضافه شدن شاخص بلوک صفر به صورت دائم، از محوشدگی ارتباط توکن‌های متنی با بردار تجمیع‌کننده نهایی جلوگیری می‌نماید. درون بلوک‌های انتخاب‌شده، ماتریس پرتاب گلوگاهی رتبه-پایین $Z = X W_{\text{down}} \in \mathbb{R}^{B \times N \times r}$ با رتبه $r \ll D$ اعمال شده و سپس به فضاهای هدهای چندگانه گسترش می‌یابد:
\begin{equation*}
Q, K, V = \operatorname{ExpandHeads}(Z W_{\text{up}}) \in \mathbb{R}^{B \times H \times N \times d_k}
\end{equation*}
توجه مقیاس‌شده صرفاً روی اشتراک بلوک‌های گزینش‌شده محاسبه شده و در نهایت پس از سنتز بافت، پرتاب خطی خروجی بردار پنهان نهایی را تولید می‌کند:
\begin{equation*}
Y = \operatorname{Context}(Q, K_{\mathcal{I}}, V_{\mathcal{I}}) W_{\text{out}} + X
\end{equation*}

\textbf{تحلیل کارایی و پیچیدگی:}
این تفکیک بلوکی و فاکتورسازی دوگانه، پیچیدگی محاسباتی توجه را از $\mathcal{O}(N^2)$ به $\mathcal{O}(N \cdot k \cdot B_s)$ کاهش می‌دهد. در آزمون‌های تجربی روی دادگان \lr{SST-2}، این ماژول موفق به دستیابی به دقت ۸۸.۶٪ شده است که نشان‌دهنده حفظ حداکثری ظرفیت بازنمایی مدل در عین کاهش چشمگیر زمان استنتاج و مصرف حافظه است.
""",

    # 5. galerkin_attention
    r"""
\subsection*{مکانیزم توجه تصویرسازی پتروف-گالرکین (\lr{Petrov-Galerkin Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه پتروف-گالرکین (\lr{Galerkin Attention})}

\textbf{مبانی نظری و انگیزه معماری:}
برخلاف دیدگاه سنتی که مکانیزم توجه را یک نگاشت گسسته میان بردارها تلقی می‌کند، معماری توجه پتروف-گالرکین (\lr{Petrov-Galerkin Projection Attention}) خودتوجهی را به عنوان یک عملگر پیوسته انتگرالی فردهولم (\lr{Non-local Fredholm Integral Operator}) بر روی فضاهای تابعی فرموله‌بندی می‌کند:
\begin{equation*}
\mathcal{K} v(x) = \int \kappa(x, y) v(y) \, dy
\end{equation*}
در این چارچوب، گسسته‌سازی هسته انتگرالی از طریق تصویرسازی متعامد پتروف-گالرکین بر روی یک پایه ارتونرمال $r$-بعدی $\Phi \in \mathbb{R}^{N \times r}$ که در شرط $\Phi^\top \Phi = I_r$ صدق می‌کند صورت می‌گیرد. با استفاده از پایه‌های طیفی مشتق‌شده از تبدیل کسینوسی گسسته نوع دوم (\lr{DCT-II})، این معماری کل انرژی طیفی تنسورهای کلید و مقدار را حفظ کرده و مانع از فروپاشی رتبه (\lr{Rank Collapse}) در تقریب‌های رتبه-پایین می‌گردد.

\textbf{فرمول‌بندی و روابط ریاضی:}
ماتریس پایه ارتونرمال تبدیل کسینوسی $\Phi$ به صورت تحلیلی به شرح زیر تعریف می‌شود:
\begin{equation*}
\Phi_{n, k} = \begin{cases} 
\frac{1}{\sqrt{N}} & k = 0 \\ 
\sqrt{\frac{2}{N}} \cos\left(\frac{\pi k (2n + 1)}{2N}\right) & 1 \le k < r 
\end{cases}
\end{equation*}
تنسورهای کلید و مقدار ورودی $K, V \in \mathbb{R}^{B \times H \times N \times d}$ از طریق ضرب در این پایه طیفی به زیرفضای فشرده ارتونرمال تصویر می‌شوند:
\begin{equation*}
K_\Phi = \Phi^\top K \in \mathbb{R}^{B \times H \times r \times d}, \quad V_\Phi = \Phi^\top V \in \mathbb{R}^{B \times H \times r \times d}
\end{equation*}
سپس ماتریس امتیازهای توجه گسسته‌سازی‌شده نسبت به پایه‌های متعامد محاسبه می‌گردد:
\begin{equation*}
S = \frac{Q K_\Phi^\top}{\sqrt{d}} \in \mathbb{R}^{B \times H \times N \times r}
\end{equation*}
\begin{equation*}
P = \operatorname{Softmax}(S, \text{dim}=-1) \in \mathbb{R}^{B \times H \times N \times r}
\end{equation*}
بازنمایی بازسازی‌شده نهایی از اعمال توزیع احتمال بر مقادیر تصویرشده حاصل می‌شود:
\begin{equation*}
O = P V_\Phi \in \mathbb{R}^{B \times H \times N \times d}
\end{equation*}
از منظر تحلیلی، چون عملگر $\Phi \Phi^\top$ یک عملگر تصویر متعامد با رتبه $r$ است، خطای تقریب ماتریس توجه نسبت به توجه استاندارد دارای کران تحلیلی فروبنیوس زیر خواهد بود:
\begin{equation*}
\|A V - P V_\Phi\|_F \le \mathcal{O}\left( \sum_{k=r+1}^N \sigma_k(K) \right)
\end{equation*}
که نشان‌دهنده همگرایی یکنواخت تقریب با افزایش رتبه طیفی $r$ است.

\textbf{سازوکار حافظه و تحلیل پیچیدگی:}
این ساختار نوآورانه، مرتبه پیچیدگی زمانی و حافظه را از مقدار مربعی $\mathcal{O}(N^2 \cdot d)$ به مقیاس خطی $\mathcal{O}(N \cdot r \cdot d)$ تقلیل می‌دهد. به دلیل رتبه $r \ll N$، نیازی به نگهداری ماتریس کامل $N \times N$ نیست و ذخیره‌سازی موقت به اندازه بسیار کوچک $N \times r$ کاهش می‌یابد که این امر مانع از سرریز حافظه در پردازش توالی‌های طولانی متنی می‌گردد.
""",

    # 6. gated_delta_net
    r"""
\subsection*{شبکه دلتا گیت‌دار چندسره (\lr{Gated DeltaNet})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری شبکه \lr{Gated DeltaNet}}

\textbf{مبانی نظری و انگیزه معماری:}
ماژول \lr{Gated DeltaNet} توجه خطی را به عنوان یک حافظه تداعی‌گر ماتریسی بازگشتی با قابلیت خود-اصلاحی فرموله‌بندی می‌کند. در مدل‌های مرسوم توجه خطی (نظیر \lr{Linear Attention})، انباشت ساده ماتریس‌های ضرب خارجی کلید-مقدار منجر به اشباع سریع حافظه و افت توانایی تفکیک توکن‌ها در فواصل طولانی می‌شود. \lr{Gated DeltaNet} این معضل بنیادین را با پیاده‌سازی قاعده کلاسیک دلتا (\lr{Delta Rule}) برطرف می‌سازد. در این قانون، پیش از ذخیره اطلاعات جدید، میزان خطای پیش‌بینی مقدار فعلی توسط حالت پیشین محاسبه شده و تنها مقدار تفاضلی (\lr{Error Correction}) به حافظه افزوده می‌شود. علاوه بر این، جهت پشتیبانی کامل از مدل‌های دوطرفه نظیر \lr{DistilBERT}، اسکن بازگشتی به صورت دوسویه (\lr{Bidirectional}) پیاده‌سازی شده است.

\textbf{فرمول‌بندی و روابط ریاضی:}
در هر گام زمانی $t$، نگاشت‌های ورودی و نرخ یادگیری محلی وابسته به ورودی $\beta_t \in (0, 1)$ و ضریب واپاشی $\lambda_t \in (0, 1)$ محاسبه می‌شوند:
\begin{equation*}
q_t = X_t W_q, \quad k_t = X_t W_k, \quad v_t = X_t W_v
\end{equation*}
\begin{equation*}
\beta_t = \sigma(X_t W_\beta + b_\beta), \quad \lambda_t = \sigma(X_t W_\lambda + b_\lambda)
\end{equation*}
بروزرسانی حالت پنهان ماتریسی در جهت مستقیم بر پایه قانون اصلاح خطای دلتا به شرح زیر صورت می‌گیرد:
\begin{equation*}
\vec{S}_t = \lambda_t \odot \vec{S}_{t-1} + \beta_t \left( v_t - \vec{S}_{t-1} k_t \right) k_t^\top \in \mathbb{R}^{H \times d_v \times d_k}
\end{equation*}
به طور همزمان، اسکن پس‌رو برای تجمیع اطلاعات آینده در مدل‌های رمزگذار اجرا می‌گردد:
\begin{equation*}
\overleftarrow{S}_t = \lambda_t \odot \overleftarrow{S}_{t+1} + \beta_t \left( v_t - \overleftarrow{S}_{t+1} k_t \right) k_t^\top \in \mathbb{R}^{H \times d_v \times d_k}
\end{equation*}
بازخوانی اطلاعات از حالت‌های پنهان متناظر با بردار پرس‌وجو به صورت زیر است:
\begin{equation*}
\vec{Y}_t = \vec{S}_t q_t, \quad \overleftarrow{Y}_t = \overleftarrow{S}_t q_t
\end{equation*}
ترکیب خروجی دو مسیر و پرتاب نهایی به شکل زیر انجام می‌پذیرد:
\begin{equation*}
Y_t = \frac{1}{2} \left( \vec{Y}_t + \overleftarrow{Y}_t \right), \quad \text{Output} = Y W_o + b_o \in \mathbb{R}^{B \times L \times D}
\end{equation*}

\textbf{سازوکار حافظه و تحلیل پیچیدگی:}
پیچیدگی زمانی این ماژول کاملاً خطی و از مرتبه $\mathcal{O}(L \cdot H \cdot d_k \cdot d_v)$ است. در حالت استنتاج خودهمبسته، بر خلاف مدل‌های سنتی که نیازمند کش ذخیره‌سازی متناوب $\mathcal{O}(L)$ هستند، وضعیت سیستم منحصراً درون ماتریس حالت ثابت $\mathcal{O}(d_k \cdot d_v)$ نگهداری می‌شود. این ویژگی جهش عظیمی در سرعت پردازش و کاهش بار مصرفی حافظه به بار می‌آورد.
""",

    # 7. gated_deltanet_block
    r"""
\subsection*{بلوک هیبرید توجه چندسره و دلتانت گیت‌دار (\lr{Gated DeltaNet Block})}
\phantomsection
\addcontentsline{toc}{subsection}{بلوک هیبرید \lr{Gated DeltaNet Block}}

\textbf{مبانی نظری و انگیزه معماری:}
\lr{Gated DeltaNet Block} یک بلوک معماری انعطاف‌پذیر و هیبرید است که پل رابطی میان مکانیزم توجه استاندارد با دقت بالا و ساختارهای بازگشتی خطی فوق‌سریع برقرار می‌کند. در پردازش زبان طبیعی، لایه‌های اولیه شبکه غالباً نیازمند دقت بالا در دریافت جزییات محلی هستند، در حالی که لایه‌های عمیق‌تر از مزایای سرعت و حافظه خطی بهره‌مند می‌شوند. این بلوک با دارا بودن یک سوییچ تغییر وضعیت دوگانه (\lr{Dual-Mode Operation})، امکان جابجایی بدون درز میان حالت توجه کامل چندسره (\lr{Full Attention Mode}) در مراحل راه‌اندازی و آموزش اولیه و حالت دلتانت گیت‌دار خطی (\lr{Recurrent Linear Mode}) برای استنتاج بلادرنگ روی توالی‌های عظیم را فراهم می‌آورد.

\textbf{فرمول‌بندی و روابط ریاضی:}
ورودی پنهان $X$ از طریق پرتاب فاکتورسازی‌شده گلوگاهی رتبه-پایین به ابعاد میانی هدایت می‌شود:
\begin{equation*}
Z = X W_{\text{down}} \in \mathbb{R}^{B \times N \times r}, \quad r \ll D
\end{equation*}
سپس شاخه‌های محاسباتی بر اساس حالت عملیاتی پیکربندی می‌شوند:
\begin{itemize}
    \item \textbf{در حالت توجه کامل:} خروجی بر اساس ضرب داخلی کلاسیک و نرمال‌سازی سافت‌مکس محاسبه می‌گردد:
    \begin{equation*}
    Y_{\text{attn}} = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
    \end{equation*}
    \item \textbf{در حالت دلتانت خطی:} محاسبات بر اساس اسکن خطی بازگشتی متکی بر قاعده دلتا و گیتینگ دوسویه صورت می‌پذیرد:
    \begin{equation*}
    Y_{\text{deltanet}} = \operatorname{RecurrentDeltaScan}(Q, K, V, \lambda, \beta)
    \end{equation*}
\end{itemize}
تلفیق نهایی شاخه‌ها به کمک یک گیت سیگموئیدی پویا کنترل می‌گردد:
\begin{equation*}
g = \sigma(X W_{\text{gate}} + b_{\text{gate}}), \quad Y = g \odot Y_{\text{attn}} + (1 - g) \odot Y_{\text{deltanet}}
\end{equation*}
\begin{equation*}
\text{Output} = Y W_{\text{out}} + X
\end{equation*}

\textbf{سازوکار حافظه و کارایی:}
این معماری تطبیقی، در سناریوهای استنتاج توالی‌های طولانی هزینه را از مرتبه $\mathcal{O}(L^2)$ به مقیاس خطی $\mathcal{O}(L)$ کاهش داده و اندازه وضعیت پنهان بازگشتی را به ابعاد ثابت $H \times d_k \times d_v$ محدود می‌کند. این راهبرد تعادل بی‌نقصی میان دقت بازنمایی زبانی و کارایی پردازش حافظه پدید می‌آورد.
"""
]
print("Chunk 1 loaded successfully, total modules:", len(modules_chunk1))
