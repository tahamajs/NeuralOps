with open('thesis_final/master_thesis.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Add Loc 1 if not present
if 'fig:base_mha_pipeline' not in text:
    old_target1 = r'''در این معادله، $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_k}$ ماتریس‌های ویژگی توکن‌ها هستند. حاصل‌ضرب $\mathbf{Q} \mathbf{K}^\top$ یک ماتریس مربعی $N \times N$ تولید می‌کند که پیچیدگی زمانی و حافظه‌ای لایه را به شدت مقید به طول کانتکست می‌سازد.'''
    
    new_text1 = old_target1 + r'''

شکل \ref{fig:base_mha_pipeline} خط لوله و گردش تانسوری لایه توجه چندسره مقیاس‌شده \lr{(Multi-Head Scaled Dot-Product Attention)} را به همراه سازوکارهای مدرن انطباق کم‌مرتبه \lr{(LoRA)}، جاسازی موقعیتی چرخشی \lr{(RoPE)} و تکرار سرهای کلید-مقدار در رویکرد \lr{GQA} نمایش می‌دهد که به عنوان معماری مبنا در این رساله مد نظر قرار گرفته است.

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{img/arch_full_attention.jpeg}
\caption{معماری لایه توجه چندسره استاندارد مقیاس‌شده به همراه سازوکارهای \lr{LoRA} و \lr{RoPE}}
\label{fig:base_mha_pipeline}
\end{figure}'''

    if old_target1 in text:
        text = text.replace(old_target1, new_text1)
        print('Location 1 updated successfully!')
    else:
        print('Location 1 target NOT found!')

# Add Loc 2 if not present
if 'fig:champion_fusion_detailed' not in text:
    old_target2 = r'''ماژول \lr{Fusion Attention} با ادغام دو استراتژی پیشرفته ابداع گردیده است: قطعه‌بندی بلاکی تانسورها \lr{(Block-Sparse Tiling)} با انتخاب پویا بر اساس امتیازات میانگین سنتروید بلاک‌ها، همراه با تجزیه مقادیر منفرد بر روی ماتریس تصویرسازی مقادیر $(\mathbf{V})$. بدین ترتیب، تنها بلاک‌های پرس‌وجو و کلید با همبستگی بالا ضرب شده و عملیات پروجکشن ارزش با ماتریس کم‌مرتبه انجام می‌پذیرد.'''
    
    new_text2 = old_target2 + r'''

همان‌طور که در شکل \ref{fig:champion_fusion_detailed} مشخص است، این معماری از یک مسیریاب هوشمند طول دنباله \lr{(Router)} بهره می‌گیرد؛ به گونه‌ای که توالی‌های کوتاه \lr{($N \le 128$)} از مسیر سریع استاندارد \lr{(Fast-Path)} عبور می‌کنند و برای توالی‌های طولانی‌تر \lr{($N > 128$)}، قطعه‌بندی بلاکی با ابعاد ۶۴ توکن همراه با استخر میانگین، شبکه گیتینگ \lr{MLP} و انتخاب تنک ۴ بلاک برتر \lr{(Top-K = 4)} فعال می‌گردد تا پیچیدگی محاسباتی در طول‌های بلند به طور خطی مقیاس‌پذیر بماند.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{img/arch_fusion_attn.jpeg}
\caption{معماری تفصیلی مدل قهرمان \lr{Fusion Attention} با تفکیک مسیر سریع توالی‌های کوتاه و توجه تنک بلاکی توالی‌های بلند}
\label{fig:champion_fusion_detailed}
\end{figure}'''

    if old_target2 in text:
        text = text.replace(old_target2, new_text2)
        print('Location 2 updated successfully!')
    else:
        print('Location 2 target NOT found!')

with open('thesis_final/master_thesis.tex', 'w', encoding='utf-8') as f:
    f.write(text)
