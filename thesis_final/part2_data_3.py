# -*- coding: utf-8 -*-

modules_chunk3 = [
    # 15. hgrn2_gated_deltanet
    r"""
\subsection*{معماری ترکیبی \lr{HGRN-2 Gated DeltaNet}}
\phantomsection
\addcontentsline{toc}{subsection}{معماری ترکیبی \lr{HGRN-2 Gated DeltaNet}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{HGRN-2 Gated DeltaNet} یک ساختار هیبرید پیشرفته از دو پارادایم بازگشتی خطی مکمل است:
\begin{enumerate}
    \item \textbf{شاخه درایه‌به‌درایه \lr{HGRN-2}:} یک شبکه بازگشتی با گیتینگ سلسله‌مراتبی که با پیچیدگی بسیار پایین $\mathcal{O}(L \cdot D)$، پویایی‌های محلی با بسامد بالا و روابط همسایگی در توالی را به سرعت مدل‌سازی می‌کند.
    \item \textbf{شاخه حافظه تداعی‌گر \lr{Gated DeltaNet}:} یک ساختار ماتریسی چندسره که از قاعده یادگیری دلتا برای نگهداری، بازیابی و تصحیح اطلاعات در یک حافظه تداعی‌گر سراسری بهره می‌برد.
\end{enumerate}
تلفیق این دو دیدگاه امکان مدل‌سازی توأمان الگوهای زبانی کوتاه‌مدت و بلندمدت را در قالب یک سازوکار کاملاً خطی فراهم می‌آورد.

\textbf{فرمول‌بندی و روابط ریاضی:}
ورودی پنهان $X$ به طور همزمان به هر دو شاخه تزریق می‌شود:
\begin{align*}
Y_{\text{hgrn}} &= \operatorname{HGRN2}(X) \in \mathbb{R}^{B \times L \times D} \\
Y_{\text{deltanet}} &= \operatorname{GatedDeltaNet}(X) \in \mathbb{R}^{B \times L \times D}
\end{align*}
جهت ترکیب بهینه این دو بازنمایی، یک گیت سیگموئیدی پویا و وابسته به محتوای ورودی محاسبه می‌شود:
\begin{equation*}
\gamma = \sigma(X W_g + b_g) \in (0, 1)^{B \times L \times D}
\end{equation*}
سپس بردار نهایی از طریق ترکیب محدب دو شاخه و اعمال پرتاب خطی استخراج می‌گردد:
\begin{equation*}
Y = \gamma \odot Y_{\text{hgrn}} + (1 - \gamma) \odot Y_{\text{deltanet}}
\end{equation*}
\begin{equation*}
\text{Output} = Y W_o + b_o \in \mathbb{R}^{B \times L \times D}
\end{equation*}

\textbf{سازوکار حافظه و تحلیل پیچیدگی:}
پیچیدگی محاسباتی زمانی کل سیستم برابر با $\mathcal{O}(B \cdot L \cdot (D + H \cdot d_k \cdot d_v))$ است که وابستگی به طول توالی را در سطح کاملاً خطی نگه می‌دارد. وضعیت پنهان مورد نیاز در طول استنتاج ترکیبی از بردار اندازه $D$ برای شاخه \lr{HGRN-2} و ماتریس $H \times d_k \times d_v$ برای شاخه دلتانت است که در مقایسه با انباشت نمایی کش در مدل‌های متراکم، حافظه ایستا و ثابتی را مصرف می‌نماید.
""",

    # 16. hgrn2_layer
    r"""
\subsection*{لایه شبکه بازگشتی سلسله‌مراتبی گیت‌دار نسخه ۲ (\lr{HGRN-2 Layer})}
\phantomsection
\addcontentsline{toc}{subsection}{لایه \lr{HGRN-2 Layer}}

\textbf{مبانی نظری و انگیزه معماری:}
لایه \lr{HGRN-2 (Hierarchically Gated Recurrent Network V2)} پاسخی پیشرفته به چالش موازی‌سازی و راندمان محاسباتی در شبکه‌های بازگشتی است. این مدل مکانیزم توجه را با گیت‌های خطی درایه‌به‌درایه و توابع واپاشی وابسته به حالت جایگزین می‌سازد. در این ساختار، با تفکیک ابعاد پنهان و اجتناب از ضرب‌های ماتریسی پرهزینه در طول گام‌های زمانی، امکان استفاده از الگوریتم‌های اسکن تجمعی انجمنی (\lr{Associative Scan}) در سطح سخت‌افزار فراهم می‌گردد که آموزش موازی و استنتاج گام‌به‌گام را به حداکثر سرعت می‌رساند.

\textbf{فرمول‌بندی و روابط ریاضی:}
در هر توکن $x_t \in \mathbb{R}^D$، گیت فراموشی وابسته به داده و بردار تزریق ورودی محاسبه می‌شوند:
\begin{equation*}
f_t = \sigma(W_f x_t + b_f) \quad \text{یا} \quad f_t = \exp\left(-\operatorname{softplus}(W_f x_t)\right) \in (0, 1)^D
\end{equation*}
\begin{equation*}
i_t = \tanh(W_i x_t + b_i) \in [-1, 1]^D
\end{equation*}
بروزرسانی حالت بازگشتی پنهان به صورت درایه‌به‌درایه از رابطه زیر تبعیت می‌کند:
\begin{equation*}
h_t = f_t \odot h_{t-1} + (1 - f_t) \odot i_t \in \mathbb{R}^D
\end{equation*}
مدولاسیون غیرخطی خروجی با استفاده از تابع فعال‌سازی خودگیتینگ \lr{SiLU} انجام می‌پذیرد:
\begin{equation*}
y_t = h_t \odot \operatorname{SiLU}(W_o x_t) \in \mathbb{R}^D
\end{equation*}
\begin{equation*}
\text{Output}_t = y_t W_{\text{proj}} + b_{\text{proj}}
\end{equation*}

\textbf{تحلیل حافظه و سخت‌افزار:}
پیچیدگی محاسباتی زمانی مدل به ازای توالی با طول $L$ دقیقاً برابر با $\mathcal{O}(L \cdot D)$ است که حداقل پیچیدگی نظری ممکن برای پردازش دنباله‌ها محسوب می‌شود. در استنتاج، ردپای وضعیت پنهان برابر با یک بردار ساده $\mathcal{O}(D)$ است. پیاده‌سازی این عملگر با کرنل‌های همجوش‌یافته اختصاصی در پردازنده‌های گرافیکی، توان عملیاتی پردازش توکن بر ثانیه را به شدت افزایش می‌دهد.
""",

    # 17. hgrn2_outer
    r"""
\subsection*{توجه خطی بازگشتی ضرب خارجی \lr{HGRN-2 Outer-Product}}
\phantomsection
\addcontentsline{toc}{subsection}{توجه ضرب خارجی \lr{HGRN-2 Outer}}

\textbf{مبانی نظری و انگیزه معماری:}
اگرچه لایه‌های بازگشتی درایه‌به‌درایه نظیر \lr{HGRN-2} سرعت اجرایی فوق‌العاده‌ای دارند، ظرفیت ذخیره‌سازی اطلاعات تداعی‌گر آن‌ها به دلیل استفاده از یک بردار تک‌بعدی محدود است. ماژول \lr{HGRN-2 Outer-Product} برای ارتقای ظرفیت بیانی، اصل بازگشت سلسله‌مراتبی را به فضای ماتریسی از طریق ضرب خارجی بردار کلید و مقدار ($K_t \otimes V_t$) بسط می‌دهد. این فرآیند یک ماتریس حافظه مرتبه دو تشکیل می‌دهد که توانایی کدگذاری روابط پیچیده چندسره را بدون قربانی کردن مزیت خطی بودن تضمین می‌نماید.

\textbf{فرمول‌بندی و روابط ریاضی:}
با فرض هدهای چندگانه، تنسورهای پرس‌وجو، کلید و مقدار و ضرایب واپاشی استخراج می‌شوند:
\begin{equation*}
Q_t, K_t, V_t \in \mathbb{R}^{H \times d_k}, \quad \lambda_t = \sigma(X_t W_\lambda + b_\lambda) \in (0, 1)^{H \times 1 \times 1}
\end{equation*}
ماتریس حالت بازگشتی از طریق افزودن حاصل‌ضرب خارجی کلید و مقدار به حافظه قبلی تشکیل می‌گردد:
\begin{equation*}
\mathbf{S}_t = \lambda_t \odot \mathbf{S}_{t-1} + K_t V_t^\top \in \mathbb{R}^{H \times d_k \times d_v}
\end{equation*}
همچنین متغیر ردیاب مخرج جهت حفظ پایداری عددی به شرح زیر اسکن می‌شود:
\begin{equation*}
\mathbf{z}_t = \lambda_t \odot \mathbf{z}_{t-1} + K_t \in \mathbb{R}^{H \times d_k}
\end{equation*}
بازخوانی بردار زمینه از ضرب ماتریس حالت در بردار پرس‌وجو انجام می‌پذیرد:
\begin{equation*}
Y_t = \frac{\mathbf{S}_t^\top Q_t}{\mathbf{z}_t^\top Q_t + \epsilon} \in \mathbb{R}^{H \times d_v}
\end{equation*}
سپس هدهای مختلف الحاق شده و پرتاب خروجی بردار نهایی را می‌سازد:
\begin{equation*}
\text{Output}_t = \operatorname{Concat}(Y_{t, 1}, \dots, Y_{t, H}) W_{\text{out}} + b_{\text{out}}
\end{equation*}

\textbf{تحلیل کارایی:}
پیچیدگی محاسباتی زمانی مدل $\mathcal{O}(L \cdot H \cdot d_k \cdot d_v)$ است که با طول توالی $L$ مقیاس خطی دارد. ظرفیت ذخیره‌سازی ماتریسی این امکان را فراهم می‌آورد که روابط نحوی پیچیده و وابستگی‌های عمیق معنایی بدون اشباع زودرس در حافظه ثبت شوند.
""",

    # 18. hla_attention
    r"""
\subsection*{توجه خطی هادامارد با نگاشت‌های نامنفی \lr{HLA}}
\phantomsection
\addcontentsline{toc}{subsection}{توجه خطی هادامارد (\lr{Hadamard Linear Attention})}

\textbf{مبانی نظری و انگیزه معماری:}
معماری توجه خطی هادامارد (\lr{Hadamard Linear Attention - HLA})، ماتریس پرهزینه سافت‌مکس $\mathcal{O}(L^2)$ را با یک سیستم حافظه بازگشتی خطی و چندسره جایگزین می‌کند. ویژگی شاخص \lr{HLA} استفاده از نگاشت ویژگی نامنفی هموار $\operatorname{ELU}+1$، مخرج‌های پایدار ردیاب مجموع و واپاشی هادامارد آموزش‌پذیر $\lambda \in (0, 1)$ است که به همراه اسکن دوسویه (\lr{Bidirectional Scanning})، یکپارچه‌سازی بافت کامل متن را برای مدل‌های مبتنی بر انکودر میسر می‌سازد.

\textbf{فرمول‌بندی و روابط ریاضی:}
نگاشت‌های غیرمنفی با اعمال لایه‌های پرتاب و تبدیل هسته‌ای تولید می‌گردند:
\begin{equation*}
\phi(Q) = \operatorname{ELU}(X W_q) + 1, \quad \phi(K) = \operatorname{ELU}(X W_k) + 1, \quad V = X W_v
\end{equation*}
با در نظر گرفتن ضریب واپاشی آموزش‌پذیر $\lambda = \sigma(\beta) \in (0, 1)$، معادلات بازگشتی رو به جلو و رو به عقب تدوین می‌شوند:
\begin{align*}
\vec{S}_t &= \lambda \vec{S}_{t-1} + \phi(K_t) \otimes V_t, \quad &\vec{z}_t &= \lambda \vec{z}_{t-1} + \phi(K_t) \\
\overleftarrow{S}_t &= \lambda \overleftarrow{S}_{t+1} + \phi(K_t) \otimes V_t, \quad &\overleftarrow{z}_t &= \lambda \overleftarrow{z}_{t+1} + \phi(K_t)
\end{align*}
بردارهای زمینه از نرمال‌سازی کسری در هر دو جهت استخراج می‌گردند:
\begin{equation*}
\vec{Y}_t = \frac{\vec{S}_t \phi(Q_t)}{\vec{z}_t^\top \phi(Q_t) + \epsilon}, \quad \overleftarrow{Y}_t = \frac{\overleftarrow{S}_t \phi(Q_t)}{\overleftarrow{z}_t^\top \phi(Q_t) + \epsilon}
\end{equation*}
خروجی نهایی از ترکیب میانگین دو جریان و پرتاب خطی حاصل می‌شود:
\begin{equation*}
Y = \frac{1}{2} \left( \vec{Y} + \overleftarrow{Y} \right), \quad \text{Output} = Y W_{\text{out}} + b_{\text{out}} \in \mathbb{R}^{B \times L \times D}
\end{equation*}

\textbf{سازوکار حافظه و تحلیل پیچیدگی:}
پیچیدگی زمانی محاسبات با طول توالی کاملاً خطی $\mathcal{O}(L \cdot d^2)$ است. مخرج‌های ردیاب $\vec{z}_t$ مانع از ناپایداری‌های عددی تقسیم بر صفر شده و نگاشت نامنفی $\phi(\cdot)$ تضمین می‌کند که بردارهای پنهان همواره در مخروط محدب معتبر فضا قرار داشته باشند.
""",

    # 19. hymba_transx
    r"""
\subsection*{بلوک هیبرید ترانسفورمر دیفرانسیلی و مدل فضای حالت هیمبا (\lr{HymbaTransX})}
\phantomsection
\addcontentsline{toc}{subsection}{بلوک هیبرید \lr{HymbaTransX}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{HymbaTransX} یک بلوک ترکیبی پیشرفته است که دو نوآوری برجسته در زمینه پردازش متن را پیوند می‌دهد:
\begin{enumerate}
    \item \textbf{توجه دیفرانسیلی (\lr{Differential Attention}):} این مکانیزم با بکارگیری دو نگاشت موازی سافت‌مکس و تفاضل آن‌ها از یکدیگر، نویزهای پس‌زمینه حالت مشترک (\lr{Common-Mode Noise}) را حذف کرده، مانع از تشکیل پدیده‌های چاه توجه (\lr{Attention Sinks}) شده و مسیریابی معنایی میان توکن‌ها را به شدت متمرکز و شفاف می‌سازد.
    \item \textbf{مدل بازگشتی فضای حالت دوسویه (\lr{Bidirectional SimpleSSM}):} یک شاخه همزمان فضای حالت که اطلاعات سراسری توالی را با پیچیدگی خطی در کل طول متن منتشر می‌نماید.
\end{enumerate}

\textbf{فرمول‌بندی و روابط ریاضی:}
پرس‌وجوها و کلیدها با ماتریس‌های دوران مکانی \lr{RoPE} چندسره کدگذاری می‌شوند:
\begin{equation*}
\mathbf{R}_{\Theta, m}^{d_k} \mathbf{x}_m = \begin{pmatrix} x_{m, 0} \cos(m \theta_0) - x_{m, 1} \sin(m \theta_0) \\ x_{m, 0} \sin(m \theta_0) + x_{m, 1} \cos(m \theta_0) \\ \vdots \end{pmatrix}
\end{equation*}
سپس جفت‌های دوگانه پرس‌وجو-کلید $(\mathbf{Q}_1, \mathbf{K}_1)$ و $(\mathbf{Q}_2, \mathbf{K}_2)$ تشکیل می‌گردند:
\begin{equation*}
A_1 = \operatorname{Softmax}\left(\frac{Q_1 K_1^\top}{\sqrt{d_k}}\right), \quad A_2 = \operatorname{Softmax}\left(\frac{Q_2 K_2^\top}{\sqrt{d_k}}\right)
\end{equation*}
ماتریس توجه دیفرانسیلی با اعمال تفاضل مقیاس‌شده و فیلتر غیرمنفی به صورت زیر مشتق می‌شود:
\begin{equation*}
A_{\text{diff}} = \frac{\operatorname{ReLU}(A_1 - \lambda_{\text{diff}} A_2)}{\sum \operatorname{ReLU}(A_1 - \lambda_{\text{diff}} A_2) + \epsilon}, \quad A_{\text{out}} = A_{\text{diff}} V
\end{equation*}
به طور همزمان، بردار خروجی شاخه فضای حالت $M_{\text{out}} = \operatorname{SimpleSSM}(X)$ محاسبه می‌شود. ادغام نهایی از طریق یک گیت تعادلی آلفا صورت می‌پذیرد:
\begin{equation*}
Y = \alpha \odot A_{\text{out}} + (1 - \alpha) \odot M_{\text{out}}
\end{equation*}
\begin{equation*}
\text{Output} = Y W_{\text{out}} + b_{\text{out}}
\end{equation*}

\textbf{تحلیل کارایی و ویژگی‌های متمایز:}
حذف نویز در شاخه دیفرانسیلی دقت تفکیک معانی نزدیک را افزایش می‌دهد، در حالی که شاخه \lr{SSM} بار محاسباتی توالی‌های بسیار طولانی را مهار می‌کند. این معماری هم‌افزایی بی‌نظیری میان تمرکز متنی و مقیاس‌پذیری سخت‌افزاری ایجاد می‌نماید.
""",

    # 20. infini_attention
    r"""
\subsection*{مکانیزم توجه نامحدود اینفینی (\lr{Infini-Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری توجه \lr{Infini-Attention}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{Infini-Attention} که توسط پژوهشگران گوگل معرفی شد، امکان بسط پنجره زمینه مدل‌های ترانسفورمر را به توالی‌های بی‌نهایت طولانی بدون انفجار مصرف حافظه فراهم می‌آورد. ایده بنیادین این ماژول، تلفیق توجه موضعی پنجره لغزان مبتنی بر نقطه‌ای-مقیاس‌شده با یک حافظه فشرده‌ساز سراسری پیوسته (\lr{Compressive Memory}) در هر هد توجه است. به جای دور انداختن توکن‌های گذشته پس از خروج از پنجره محلی، این اطلاعات به درون یک ماتریس حافظه با بعد ثابت فشرده می‌شوند و در زمان نیاز مجدداً بازیابی می‌گردند.

\textbf{فرمول‌بندی و روابط ریاضی:}
در هر بخش متنی، خروجی ماژول حاصل ترکیب دو جریان محلی و فشرده است:
\begin{equation*}
\mathbf{Y} = \mathbf{Y}_{\text{local}} + \sigma(\beta) \odot \mathbf{Y}_{\text{mem}}
\end{equation*}
\begin{enumerate}
    \item \textbf{جریان محلی پنجره لغزان:}
    \begin{equation*}
    \mathbf{Y}_{\text{local}} = \operatorname{Softmax}\left(\frac{Q K_{\text{local}}^\top}{\sqrt{d}}\right) V_{\text{local}}
    \end{equation*}
    \item \textbf{جریان حافظه فشرده‌ساز تداعی‌گر:}
    ماتریس حافظه با استفاده از نگاشت‌های فعال‌سازی $\sigma(\cdot)$ با قاعده جمع خطی بروزرسانی می‌شود:
    \begin{equation*}
    \mathbf{M}_t = \mathbf{M}_{t-1} + \sigma(K_t)^\top V_t \in \mathbb{R}^{d_k \times d_v}
    \end{equation*}
    \begin{equation*}
    \mathbf{z}_t = \mathbf{z}_{t-1} + \sum_{m} \sigma(K_{t, m})^\top \in \mathbb{R}^{d_k}
    \end{equation*}
    بازیابی اطلاعات از حافظه با بردار پرس‌وجو به شرح زیر انجام می‌گیرد:
    \begin{equation*}
    \mathbf{Y}_{\text{mem}} = \frac{\sigma(Q) \mathbf{M}_t}{\sigma(Q) \mathbf{z}_t + \epsilon}
    \end{equation*}
\end{enumerate}

\textbf{تحلیل پیچیدگی و پایداری حافظه:}
ویژگی انقلابی \lr{Infini-Attention} در این است که ردپای حافظه کش برای کل تاریخچه نامحدود متن همواره در اندازه ثابت $\mathcal{O}(d_k \cdot d_v)$ در هر هد قفل می‌شود و هرگز با افزایش طول دنباله ($L \to \infty$) افزایش نمی‌یابد. در ارزیابی‌های دادگان \lr{SST-2}، پشتیبانی دوسویه از بافت توانست دقت مدل را از ۵۰.۹۲٪ به ۸۸.۳٪ بازگرداند که توانمندی خارق‌العاده آن را در حفظ همبستگی‌های دوربرد نشان می‌دهد.
""",

    # 21. jamba
    r"""
\subsection*{ترانسفورمر پیوندی جامبا-۲ (\lr{Jamba-2 Hybrid Transformer})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری پیوندی \lr{Jamba-2}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری پیوندی \lr{Jamba-2} بر پایه یکپارچه‌سازی عمیق سه فناوری بنیادین در یادگیری ماشین مقیاس‌پذیر بنا شده است:
\begin{enumerate}
    \item لایه‌های بازگشتی فضای حالت فشرده‌شده با تجزیه مقادیر منفرد (\lr{SVD-Mamba}).
    \item لایه‌های توجه متراکم چندسره (\lr{Multi-Head Attention - MHA}) جهت تثبیت پیوندهای تداعی‌گر دوربرد.
    \item لایه‌های تنک ترکیبی از متخصصان بر پایه انتخاب بهینه متخصص (\lr{Expert Choice MoE}).
\end{enumerate}
با چیدمان متناوب بلوک‌های مامبا و توجه در نسبت‌های مهندسی‌شده (نظیر ۷ بلوک مامبا به ازای هر ۱ بلوک توجه)، \lr{Jamba-2} به توان عملیاتی فوق‌العاده بالا، مصرف حافظه ناچیز و کیفیت تولید زبانی در سطح برترین مدل‌های پیشرو دست می‌یابد.

\textbf{فرمول‌بندی و روابط ریاضی:}
در لایه‌های مامبا فشرده‌شده با \lr{SVD}، ماتریس‌های پرتاب با تقریب رتبه $r$ فاکتورسازی می‌شوند:
\begin{equation*}
X_{\text{down}} = X V_{\text{svd}}, \quad Y_{\text{ssm}} = \operatorname{MambaCore}(X_{\text{down}}), \quad X_{\text{up}} = Y_{\text{ssm}} U_{\text{svd}}
\end{equation*}
در لایه‌های توجه چندسره دوره‌ای، ماتریس شباهت استاندارد محاسبه می‌گردد:
\begin{equation*}
Y_{\text{attn}} = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
\end{equation*}
در لایه‌های تنک متخصصان (\lr{MoE})، مسیریابی توکن‌ها به برترین متخصصان بر اساس انتخاب متخصص صورت می‌پذیرد تا توازن بار کاری میان واحدهای پردازشی حفظ شود:
\begin{equation*}
G = \operatorname{TopK}\left(\operatorname{Softmax}(X W_{\text{router}}), k\right)
\end{equation*}
\begin{equation*}
Y_{\text{moe}} = \sum_{e \in \mathcal{E}} G_e \cdot \operatorname{FFN}_e(X)
\end{equation*}

\textbf{تحلیل کارایی و صرفه‌جویی سخت‌افزاری:}
جایگزینی اکثر لایه‌های توجه با مدل‌های فشرده مامبا، حجم کش \lr{KV} در طول استنتاج را تا بیش از ۸۰ درصد کاهش می‌دهد. این صرفه‌جویی عظیم به مدل امکان می‌دهد که متون با طول بیش از ۲۵۶ هزار توکن را بر روی یک سرور پردازشی منفرد با سرعت بالا و تأخیر پایین پردازش کند.
"""
]
print("Chunk 3 loaded successfully, total modules:", len(modules_chunk3))
