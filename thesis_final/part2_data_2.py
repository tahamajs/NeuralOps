# -*- coding: utf-8 -*-

modules_chunk2 = [
    # 8. gated_linear_attention
    r"""
\subsection*{توجه خطی گیت‌دار با واپاشی نمایی وابسته به داده (\lr{Gated Linear Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{توجه خطی گیت‌دار (\lr{Gated Linear Attention})}

\textbf{مبانی نظری و انگیزه معماری:}
مدل‌های سنتی توجه خطی با حذف عملگر غیرخطی \lr{Softmax} و بهره‌گیری از خاصیت جابجایی‌پذیری ضرب ماتریسی به پیچیدگی خطی دست می‌یابند. با این وجود، این روش‌ها به دلیل عدم وجود مکانیزم فراموشی و تداخل اطلاعات جدید با بافت‌های گذشته دچار تضعیف ظرفیت مدل می‌شوند. معماری \lr{Gated Linear Attention (GLA)} این چالش را از طریق تلفیق دو سازوکار گیتینگ رفع می‌کند: گیت واپاشی نمایی وابسته به داده (\lr{Data-Dependent Exponential Decay Gate}) در طول بعد بازگشت و گیت غیرخطی خروجی مبتنی بر \lr{SiLU}. در ارزیابی‌های تجربی روی مدل \lr{DistilBERT} و دادگان \lr{SST-2}، این ماژول با زمان آموزش سریع ۱۹.۸ ثانیه برای هر دور، موفق به کسب دقت ۷۹.۸۲٪ و امتیاز \lr{F1} معادل ۷۹.۷۶٪ شده است.

\textbf{فرمول‌بندی و روابط ریاضی:}
تنسورهای پرس‌وجو، کلید، مقدار و گیت خروجی از طریق پرتاب‌های خطی تشکیل می‌شوند. جهت تضمین نامنفی بودن ویژگی‌ها، از نگاشت کرنلی $\phi(x) = \operatorname{ELU}(x) + 1$ استفاده می‌شود:
\begin{equation*}
Q = X W_q, \quad K = X W_k, \quad V = X W_v, \quad Z = X W_z
\end{equation*}
\begin{equation*}
\phi(Q) = \operatorname{ELU}(Q) + 1, \quad \phi(K) = \operatorname{ELU}(K) + 1
\end{equation*}
ضریب واپاشی پیوسته برای هر توکن به صورت وابسته به داده در بازه $(0, 1)$ استخراج می‌شود:
\begin{equation*}
\lambda_t = \exp\left(-\operatorname{softplus}(W_g x_t + b_g)\right) \in (0, 1)^{B \times H}
\end{equation*}
اسکن بازگشتی دوسویه برای تجمیع اطلاعات با در نظر گرفتن متغیر مخرج جهت نرمال‌سازی به صورت زیر تدوین می‌گردد:
\begin{align*}
\vec{S}_t &= \lambda_t \odot \vec{S}_{t-1} + \phi(K_t) \otimes V_t, \quad &\vec{z}_t &= \lambda_t \odot \vec{z}_{t-1} + \phi(K_t) \\
\overleftarrow{S}_t &= \lambda_t \odot \overleftarrow{S}_{t+1} + \phi(K_t) \otimes V_t, \quad &\overleftarrow{z}_t &= \lambda_t \odot \overleftarrow{z}_{t+1} + \phi(K_t)
\end{align*}
بردارهای بافت نرمال‌شده در هر دو جهت بازخوانی می‌شوند:
\begin{equation*}
\vec{Y}_t = \frac{\vec{S}_t \phi(Q_t)}{\vec{z}_t^\top \phi(Q_t) + \epsilon}, \quad \overleftarrow{Y}_t = \frac{\overleftarrow{S}_t \phi(Q_t)}{\overleftarrow{z}_t^\top \phi(Q_t) + \epsilon}
\end{equation*}
در گام پایانی، میانگین دو جهت با گیت فعال‌سازی \lr{SiLU} تلفیق شده و پرتاب خروجی اعمال می‌شود:
\begin{equation*}
Y = \frac{1}{2} (\vec{Y} + \overleftarrow{Y}), \quad \text{Output} = (Y \odot \operatorname{SiLU}(Z)) W_{\text{out}} + b_{\text{out}}
\end{equation*}

\textbf{تحلیل حافظه و کارایی سخت‌افزاری:}
پیچیدگی محاسباتی زمانی مدل به صورت قطعی $\mathcal{O}(L \cdot d^2)$ است که در مقایسه با رشد توان دوم توجه استاندارد، تسریع قابل ملاحظه‌ای در پردازش توالی‌های طولانی ایجاد می‌کند. نگهداری حافظه بافر در استنتاج به اندازه ماتریس حالت پنهان $d_k \times d_v$ محدود است و نیازی به تخصیص حافظه پویا ندارد.
""",

    # 9. goldfinch_ahn
    r"""
\subsection*{شبکه هیبرید انطباقی گلدفینچ (\lr{GoldFinch-AHN})}
\phantomsection
\addcontentsline{toc}{subsection}{شبکه هیبرید انطباقی \lr{GoldFinch-AHN}}

\textbf{مبانی نظری و انگیزه معماری:}
مدل \lr{GoldFinch-AHN} یک معماری پیوندی تطبیقی است که مکانیزم توجه پنجره لغزان محلی (\lr{Sliding Window Attention}) را با شبکه بازگشتی تداعی‌گر \lr{Gated DeltaNet} تلفیق می‌نماید. در تحلیل شکست مدل‌های اولیه خطی، مشخص گردید که استفاده از ضرب داخلی تک‌هد در بعد بالا موجب فروپاشی فضای بازنمایی زیرفضاها و افت دقت به ۵۰.۹۲٪ شده بود. معماری بازطراحی‌شده \lr{GoldFinch-AHN} این نقیصه را با احیای ساختار ۱۲ هد موازی، بازتنظیم پنجره لغزان دوسویه، بارگذاری هوشمند وزن‌های پیش‌آموزش‌دیده \lr{DistilBERT} و تخصیص بایاس مثبت اولیه ($+1.8$) در گیت تلفیق مرتفع ساخت.

\textbf{فرمول‌بندی و روابط ریاضی:}
محاسبات در دو مسیر موازی کوتاه‌مدت و بلندمدت ساماندهی می‌شوند:
\begin{enumerate}
    \item \textbf{مسیر توجه پنجره لغزان چندسره (بافت کوتاه‌مدت):}
    \begin{equation*}
    S_{i, j} = \frac{Q_i K_j^\top}{\sqrt{d_h}} + M_{\text{window}}, \quad O^{\text{short}} = \operatorname{Softmax}(S, \text{dim}=-1) V
    \end{equation*}
    که در آن $M_{\text{window}}$ ماسک نواری موضعی با پهنای باند مشخص است.
    \item \textbf{مسیر حافظه تداعی‌گر دلتانت (بافت بلندمدت):}
    \begin{equation*}
    O^{\text{long}} = \operatorname{DeltaNetRecurrence}(X)
    \end{equation*}
\end{enumerate}
ترکیب این دو مسیر به کمک یک گیت سیگموئیدی با مقداردهی اولیه پایدار صورت می‌گیرد:
\begin{equation*}
g = \sigma(X W_{\text{gate}} + 1.8) \in (0, 1)^{B \times N \times 1}
\end{equation*}
\begin{equation*}
\text{Output} = W_O \left( g \odot O^{\text{short}} + (1 - g) \odot O^{\text{long}} \right) + b_O
\end{equation*}
وجود بایاس مثبت $+1.8$ تضمین می‌کند که در ابتدای فرآیند بازآموزی، مدل به جریان پایدار توجه متراکم موضعی تکیه نموده و به تدریج با بهبود گرادیان‌ها، جریان حافظه بازگشتی بلندمدت را یکپارچه سازد.

\textbf{تحلیل کارایی و ارزیابی تجربی:}
با اعمال این تغییرات، دقت مدل بر روی دادگان \lr{SST-2} از ۵۰.۹۲٪ به ۸۳.۷۲٪ جهش یافت. پیچیدگی زمانی ماژول مرکب از $\mathcal{O}(L \cdot W \cdot d)$ برای پنجره لغزان و $\mathcal{O}(L \cdot d^2)$ برای دلتانت است که در مجموع مقیاس‌بندی کاملاً خطی با طول دنباله را به همراه دارد.
""",

    # 10. gqa_attention
    r"""
\subsection*{توجه با پرس‌وجوهای گروه‌بندی‌شده (\lr{Grouped-Query Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{توجه با پرس‌وجوهای گروه‌بندی‌شده (\lr{GQA})}

\textbf{مبانی نظری و انگیزه معماری:}
معماری توجه با پرس‌وجوهای گروه‌بندی‌شده (\lr{Grouped-Query Attention - GQA}) که توسط اینسلی و همکاران (۲۰۲۳) مطرح گردید، تعادلی بهینه میان توجه استاندارد چندسره (\lr{MHA}) و توجه چندپرس‌وجویی (\lr{Multi-Query Attention - MQA}) ایجاد می‌کند. در \lr{MHA}، هر هد پرس‌وجو دارای یک جفت هد کلید-مقدار اختصاصی است ($H_{KV} = H_Q$) که ظرفیت بیانی بالایی پدید می‌آورد اما حجم کش \lr{KV} در استنتاج خودهمبسته بسیار سنگین است. در مقابل، \lr{MQA} تمامی هدهای کلید-مقدار را به یک تک‌هد به اشتراک‌گذاشته‌شده تقلیل می‌دهد ($H_{KV} = 1$) که موجب افت کیفیت و کاهش ظرفیت یادگیری می‌گردد. \lr{GQA} هدهای پرس‌وجو را به $G = H_{KV}$ گروه مساوی با نسبت تکرار $R = H_Q / H_{KV}$ افراز می‌کند؛ به‌طوری که هر گروه از هدها یک جفت کلید-مقدار مشترک را بازخوانی می‌کنند.

\textbf{فرمول‌بندی و روابط ریاضی:}
با فرض ابعاد ورودی $X \in \mathbb{R}^{B \times N \times D}$ و بعد هر هد $d_k = D / H_Q$، ماتریس‌های پرتاب با ابعاد ساختاریافته اعمال می‌گردند:
\begin{equation*}
Q = X W_Q \in \mathbb{R}^{B \times N \times (H_Q \cdot d_k)}, \quad W_Q \in \mathbb{R}^{D \times (H_Q \cdot d_k)}
\end{equation*}
\begin{equation*}
K = X W_K \in \mathbb{R}^{B \times N \times (H_{KV} \cdot d_k)}, \quad W_K \in \mathbb{R}^{D \times (H_{KV} \cdot d_k)}
\end{equation*}
\begin{equation*}
V = X W_V \in \mathbb{R}^{B \times N \times (H_{KV} \cdot d_k)}, \quad W_V \in \mathbb{R}^{D \times (H_{KV} \cdot d_k)}
\end{equation*}
تنسورهای کلید و مقدار به ابعاد هدهای پرس‌وجو از طریق عملگر تکرار بینابینی گسترش می‌یابند:
\begin{equation*}
K_{\text{shared}} = \operatorname{repeat\_interleave}(K, R, \text{dim}=\text{head}) \in \mathbb{R}^{B \times H_Q \times N \times d_k}
\end{equation*}
\begin{equation*}
V_{\text{shared}} = \operatorname{repeat\_interleave}(V, R, \text{dim}=\text{head}) \in \mathbb{R}^{B \times H_Q \times N \times d_k}
\end{equation*}
سپس ضرب داخلی مقیاس‌شده استاندارد بر روی هدهای گسترش‌یافته محاسبه می‌شود:
\begin{equation*}
S = \frac{Q K_{\text{shared}}^\top}{\sqrt{d_k}} + M \in \mathbb{R}^{B \times H_Q \times N \times N}
\end{equation*}
\begin{equation*}
O = \operatorname{Softmax}(S, \text{dim}=-1) V_{\text{shared}} \in \mathbb{R}^{B \times H_Q \times N \times d_k}
\end{equation*}
\begin{equation*}
Y = \operatorname{Concat}(O_1, \dots, O_{H_Q}) W_O + b_O \in \mathbb{R}^{B \times N \times D}
\end{equation*}

\textbf{تحلیل حافظه و مزایای سخت‌افزاری:}
استفاده از \lr{GQA} ردپای حافظه کش \lr{KV} را دقیقا به نسبت $R$ برابر (به عنوان مثال ۴ یا ۸ برابر در مدل‌های لاما و میسترال) کاهش می‌دهد. این کاهش چشمگیر در جابجایی داده از حافظه اصلی \lr{DRAM} به پردازنده گرافیکی، امکان استفاده از اندازه‌های بزرگتر دسته داده (\lr{Batch Size}) و افزایش فوق‌العاده توان عملیاتی در استنتاج را بدون افت محسوس در دقت فراهم می‌آورد.
""",

    # 11. granite_hybrid
    r"""
\subsection*{معماری پیوندی گرانیت: تلفیق فضای حالت مامبا-۲ و توجه چندسره (\lr{Granite-Hybrid Attention})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری پیوندی گرانیت (\lr{Granite-Hybrid})}

\textbf{مبانی نظری و انگیزه معماری:}
معماری پیوندی گرانیت (\lr{Granite-Hybrid Attention}) پاسخی مهندسی و نوآورانه به محدودیت‌های ذاتی مدل‌های زبانی حجیم است. در حالی که مدل‌های فضای حالت ساختاریافته (\lr{SSM}) مانند مامبا-۲ توانایی پردازش توالی‌های طولانی با مرتبه زمانی خطی و وضعیت پنهان متراکم را دارند، در وظایف بازیابی دقیق تداعی‌گر یا پیوندهای متنی دوردست گاه با ضعف روبرو می‌شوند. معماری گرانیت با چیدمان متناوب لایه‌های \lr{Mamba-2 SSD} (\lr{State Space Duality}) و لایه‌های توجه استاندارد چندسره، مزایای هر دو پارادایم را ادغام می‌کند: پردازش موازی و سریع متن با مصرف حافظه ناچیز در لایه‌های بازگشتی و بازیابی عمیق معنایی در لایه‌های توجه.

\textbf{فرمول‌بندی و روابط ریاضی:}
لایه‌های فضای حالت بر اساس دوگانگی فضای حالت به صورت ماتریسی با ساختار بلوکی-مثلثی فرموله‌بندی می‌شوند:
\begin{equation*}
\mathbf{h}_t = \mathbf{A}_t \mathbf{h}_{t-1} + \mathbf{B}_t \mathbf{x}_t
\end{equation*}
\begin{equation*}
\mathbf{y}_t = \mathbf{C}_t \mathbf{h}_t + \mathbf{D}_t \mathbf{x}_t
\end{equation*}
به کمک تبدیل ماتریسی \lr{SSD}، این سیستم بازگشتی به عنوان یک ضرب ماتریسی نیمه‌تنک با ماسک علّی بر روی پردازنده گرافیکی موازی‌سازی می‌گردد:
\begin{equation*}
Y_{\text{ssm}} = \left( (Q K^\top) \odot \mathcal{M}_{\text{decay}} \right) V
\end{equation*}
در لایه‌های متناوب توجه چندسره، بردار پنهان برای متمرکزسازی توجه روی نشانه‌های متنی کلیدی پالایش می‌شود:
\begin{equation*}
Y_{\text{attn}} = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
\end{equation*}
به منظور تنظیم تعادل بهینه پارامترها، پرتاب‌های ورودی با استفاده از تجزیه رتبه-پایین فاکتورسازی شده و ارتباطات میان لایه‌ها از طریق اتصالات پایدار باقیمانده (\lr{Residual Connections}) پایدارسازی می‌شوند.

\textbf{سازوکار حافظه و تحلیل پیچیدگی:}
با جایگزینی بخش اعظم لایه‌های توجه با بلوک‌های فضای حالت \lr{Mamba-2}، مصرف حافظه کش در حین استنتاج به کسری ناچیز تقلیل یافته و مرتبه زمانی اجرای مدل در کل دنباله به صورت $\mathcal{O}(L \cdot D)$ مقیاس‌پذیر باقی می‌ماند، در حالی که دقت زبانی هم‌تراز با ترانسفورمرهای متراکم حفظ می‌شود.
""",

    # 12. griffin_ssd
    r"""
\subsection*{معماری ترکیبی گریفین بر پایه دوگانگی فضای حالت (\lr{Griffin SSD})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری ترکیبی گریفین (\lr{Griffin SSD})}

\textbf{مبانی نظری و انگیزه معماری:}
مدل \lr{Griffin SSD} با الهام از معماری پیشگامانه گریفین گوگل، تلفیقی کارآمد از توجه موضعی پنجره لغزان (\lr{Sliding-Window Attention}) و مدل بازگشتی دوگانگی فضای حالت (\lr{SSD Recurrence}) را ارائه می‌دهد. انگیزه کلیدی این معماری، حذف وابستگی مربعی ترانسفورمر به طول توالی و جایگزینی آن با لایه‌ای بازگشتی شبه-خطی است که قادر به ردیابی حافظه سراسری بدون انباشت حافظه کش باشد. بخش محلی مسئولیت مدلسازی دقیق روابط گرامری همسایه را برعهده دارد، در حالی که شاخه \lr{SSD} جریان معنایی دوربرد را با استفاده از پارامترهای واپاشی آموزش‌پذیر هدایت می‌کند.

\textbf{فرمول‌بندی و روابط ریاضی:}
تنسورهای پرس‌وجو، کلید و مقدار به همراه شاخص‌های \lr{SSD} استخراج می‌شوند:
\begin{equation*}
Q, K, V \in \mathbb{R}^{B \times H \times L \times d}, \quad Q_{\text{ssd}} \in \mathbb{R}^{B \times H \times L \times d}
\end{equation*}
\begin{enumerate}
    \item \textbf{شاخه پنجره لغزان محلی:}
    \begin{equation*}
    O_{\text{local}} = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + M_{\text{band}}\right) V
    \end{equation*}
    که در آن $M_{\text{band}}$ توکن‌های خارج از شعاع پنجره محلی $W$ را مسدود می‌سازد.
    \item \textbf{شاخه بازگشتی فضای حالت \lr{SSD}:}
    سیستم بازگشتی سراسری با ضریب واپاشی در فضای لگاریتمی تعریف می‌شود:
    \begin{equation*}
    \alpha_t = \exp\left(-\operatorname{softplus}(\Lambda)\right) \in (0, 1)
    \end{equation*}
    \begin{equation*}
    \mathbf{h}_t = \alpha_t \mathbf{h}_{t-1} + K_t V_t^\top \in \mathbb{R}^{d \times d}
    \end{equation*}
    \begin{equation*}
    O_{\text{ssd}} = \mathbf{h}_t Q_{\text{ssd}, t}
    \end{equation*}
\end{enumerate}
ترکیب این دو بردار از طریق گیت تطبیقی صورت می‌گیرد:
\begin{equation*}
g = \sigma(X W_g + b_g), \quad Y = g \odot O_{\text{local}} + (1 - g) \odot O_{\text{ssd}}
\end{equation*}
\begin{equation*}
\text{Output} = Y W_{\text{out}} + b_{\text{out}}
\end{equation*}

\textbf{تحلیل کارایی سخت‌افزاری:}
پیچیدگی زمانی بخش محلی $\mathcal{O}(L \cdot W \cdot d)$ و شاخه بازگشتی $\mathcal{O}(L \cdot d^2)$ است که وابستگی محاسباتی به طول دنباله $L$ را کاملاً خطی می‌سازد. این ماژول در استنتاج خودهمبسته نیاز به نگهداری کش کامل گذشته نداشته و حافظه استاتیک بسیار اندکی را مصرف می‌نماید.
""",

    # 13. halo_regla
    r"""
\subsection*{معماری هالو-رگلا: توجه خطی گیت‌دار تصحیح‌شده (\lr{HALO-ReGLA})}
\phantomsection
\addcontentsline{toc}{subsection}{معماری \lr{HALO-ReGLA}}

\textbf{مبانی نظری و انگیزه معماری:}
معماری \lr{HALO-ReGLA (Hardware-Aware Linear Optics \& Rectified Gated Linear Attention)} رویکردی مهندسی‌شده جهت رفع نواقص پایداری گرادیان در تبدیل ترانسفورمرهای متراکم به لایه‌های خطی کارآمد است. در نسخه‌های اولیه این ساختار، استفاده از تابع فعال‌سازی غیرخطی سخت \lr{ReLU} منجر به صفر شدن بیش از ۵۰٪ از فعال‌سازی‌ها، قطع جریان بازپس‌گیری گرادیان و تثبیت نامطلوب مدل در دقت تصادفی ۵۰.۹۲٪ گردیده بود. \lr{HALO-ReGLA} با اعمال یک اصلاح سه‌گانه شامل استفاده از نگاشت غیرمنفی هموار $\operatorname{ELU}+1$، تعبیه کدگذاری مکانی دورانی (\lr{RoPE}) بر روی فضاهای پرس‌وجو و کلید، و تخصیص گیت ادغام انطباقی با مقداردهی اولیه مثبت، پایداری یادگیری را تضمین نموده است.

\textbf{فرمول‌بندی و روابط ریاضی:}
ابتدا بردارهای پرس‌وجو و کلید از طریق ماتریس‌های دوران بلوکی \lr{RoPE} کدگذاری مکانی می‌شوند:
\begin{equation*}
Q_{\text{rope}} = \operatorname{RoPE}(Q), \quad K_{\text{rope}} = \operatorname{RoPE}(K)
\end{equation*}
جهت تقریب خطی هسته، نگاشت فعال‌سازی نامنفی تصحیح‌شده اعمال می‌گردد:
\begin{equation*}
\phi(x) = \operatorname{ELU}(x) + 1
\end{equation*}
محاسبه توجه خطی با بهره‌گیری از خاصیت جابجایی ضرب ماتریسی انجام می‌شود:
\begin{equation*}
O^{\text{linear}} = \frac{\phi(Q_{\text{rope}}) \left(\phi(K_{\text{rope}})^\top V\right)}{\phi(Q_{\text{rope}}) \left(\sum_{j} \phi(K_{\text{rope}})_j^\top\right) + \epsilon}
\end{equation*}
به منظور حفظ بازنمایی‌های غنی متنی در طول همگرایی، یک شاخه متراکم مجهز به گیتینگ تطبیقی با بایاس اولیه مثبت $+1.8$ در نظر گرفته شده است:
\begin{equation*}
g = \sigma(X W_{\text{gate}} + 1.8)
\end{equation*}
\begin{equation*}
\text{Output} = W_O \left( g \odot O^{\text{dense}} + (1 - g) \odot O^{\text{linear}} \right)
\end{equation*}

\textbf{ارزیابی تجربی و پیچیدگی:}
بهبودهای ساختاری فوق منجر به جهش چشمگیر دقت در دادگان \lr{SST-2} از ۵۰.۹۲٪ به ۸۳.۸۳٪ گردید. محاسبات شاخه خطی با مرتبه $\mathcal{O}(L \cdot d^2)$ اجرا شده و ساختار سخت‌افزارپسند آن امکان بهینه‌سازی گسترده در سطح کرنل‌های پردازنده گرافیکی را مهیا می‌سازد.
""",

    # 14. hamming_attention
    r"""
\subsection*{تقطیر توجه همینگ با آرام‌سازی پیوسته (\lr{Hamming Attention Distillation})}
\phantomsection
\addcontentsline{toc}{subsection}{تقطیر توجه همینگ (\lr{Hamming Attention})}

\textbf{مبانی نظری و انگیزه معماری:}
محاسبه ضرب‌های داخلی ممیز شناور در مکانیزم توجه ترانسفورمرها ($Q K^\top$) بخش عمده‌ای از توان مصرفی واحدهای محاسباتی (\lr{ALU}) و پهنای باند حافظه پردازنده گرافیکی را مصرف می‌کند. ماژول تقطیر توجه همینگ (\lr{Hamming Attention Distillation - HAD})، ضرب‌های متراکم ممیز شناور را با یک فرمول‌بندی مشتق‌پذیر از فاصله‌های همینگ (\lr{Hamming Distance}) بر روی بازنمایی‌های دوتایی‌شده علامت (\lr{Sign-Binarized Representations}) جایگزین می‌نماید. از آنجا که تابع علامت استاندارد ($\operatorname{sign}$) دارای مشتق صفر در تقریباً همه جا است و امکان آموزش مبتنی بر پس‌انتشار را سلب می‌کند، معماری \lr{HAD} از یک تقریب آرام‌سازی پیوسته تانژانت هذلولوی (\lr{Tanh Relaxation}) بهره می‌گیرد.

\textbf{فرمول‌بندی و روابط ریاضی:}
پرس‌وجوها و کلیدها با استفاده از یک تابع تانژانت هایپربولیک با پارامتر مقیاس‌پذیر شبه‌دودویی می‌شوند:
\begin{equation*}
Q_b = \tanh(3.0 \cdot \gamma_Q \odot Q) \in [-1, 1]^{B \times H \times L \times d_h}
\end{equation*}
\begin{equation*}
K_b = \tanh(3.0 \cdot \gamma_K \odot K) \in [-1, 1]^{B \times H \times L \times d_h}
\end{equation*}
که در آن $\gamma_Q, \gamma_K$ بردارهای مقیاس آموزش‌پذیر برای هر هد هستند. بر پایه قضیه همگرایی علامت، با افزایش ضریب، $\tanh(\beta x)$ به طور یکنواخت به تابع علامت میل می‌کند در حالی که مشتق تحلیلی آن همواره اکیداً مثبت باقی می‌ماند:
\begin{equation*}
\frac{d}{dx}\tanh(\beta x) = \beta \left( 1 - \tanh^2(\beta x) \right) = \beta \operatorname{sech}^2(\beta x) > 0
\end{equation*}
فاصله همینگ پیوسته میان دو بردار شبه‌دودویی به صورت تحلیلی به ضرب داخلی آن‌ها مرتبط است:
\begin{equation*}
D_H(Q_b, K_b) = \frac{d_h - Q_b K_b^\top}{2}
\end{equation*}
جهت بازیابی گستره دینامیکی لاجیت‌ها و جلوگیری از فروپاشی انتروپی احتمالات بیشینه نرمال، امتیازهای توجه با دمای کالیبره‌شده $\tau$ مقیاس‌گذاری می‌شوند:
\begin{equation*}
S = \tau \left( d_h - 2 D_H(Q_b, K_b) \right) = \tau \left( Q_b K_b^\top \right)
\end{equation*}
\begin{equation*}
P = \operatorname{Softmax}(S, \text{dim}=-1), \quad O = P V
\end{equation*}

\textbf{مزایای سخت‌افزاری و شتاب‌بخشی در استنتاج:}
در مرحله استنتاج سخت‌افزاری، بردارها به مقادیر دقیق باینری $\{-1, +1\}$ نگاشت می‌شوند و ضرب‌های ماتریسی سنگین به عملیات‌های بیتی بسیار سبک \lr{XNOR} و شمارش بیت‌های یک (\lr{POPCNT}) در سطح مدار تبدیل می‌گردند. این تحول، توان مصرفی تراشه و مساحت سیلیکون مورد نیاز را تا چندین برابر کاهش داده و کارایی فوق‌العاده‌ای در سامانه‌های نهفته به ارمغان می‌آورد.
"""
]
print("Chunk 2 loaded successfully, total modules:", len(modules_chunk2))
