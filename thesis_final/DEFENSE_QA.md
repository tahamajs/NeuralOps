# جامع‌ترین راهنمای پرسش و پاسخ جلسه دفاع رساله کارشناسی ارشد (DEFENSE_QA)
## هیئت داوران و اساتید محترم — دانشکده مهندسی برق و کامپیوتر دانشگاه تهران
**عنوان پایان‌نامه:** طراحی، ارزیابی تجربی و فشرده‌سازی مکانیزم‌های توجه خطی و تجزیه مقادیر منفرد (SVD) در مدل‌های زبانی بزرگ  
**دانشجو:** محمد طاها مجلسی (شماره دانشجویی: ۸۱۰۱۰۱۵۰۴)  
**استاد راهنما:** جناب آقای دکتر احمد کلهر  
**تاریخ دفاع:** مهرماه ۱۴۰۵  

---

## مقدمه و ساختار راهنما
این سند پاسخ‌های فوق‌العاده دقیق، منسجم و دو زبانه (فارسی و انگلیسی) به ۲۰ سوال چالشی و کلیدی اساتید داور در جلسه دفاع را به همراه ارجاع صریح به قضایای ریاضی، روابط تحلیلی، جداول تجربی و سورس‌کد پروژه ارائه می‌دهد.

---

### سوال ۱: تقطیر ۶ ستونی در برابر KL پیشرو خام (6-Pillar KD Suite vs Forward KL Divergence)
**پرسش داور:**
در جدول ارزیابی رژیم‌های تقطیر (جدول ۴.۶ رساله / جدول اسلاید ۱۰)، مشاهده می‌شود که روش تقطیر KL پیشرو خام (\lr{Forward KL}) در برخی مدل‌ها نظیر \lr{Cosformer} شباهت کسینوسی لاجیت اندکی بالاتر (0.9495 در برابر 0.9184) نسبت به بسته ۶ ستونی ثبت می‌کند. چرا چارچوب ۶ ستونی به عنوان رویکرد برتر انتخاب شده است؟ آیا بهینه‌سازی چندهدفه منجر به کاهش کیفیت شده است؟

**پاسخ به فارسی:**
خیر، این پدیده بیانگر تفاوت بنیادی میان «طمع‌ورزی محلی لاجیت‌ها» (\lr{Logit Greediness}) و «استحکام ساختاری چندلایه‌ای» (\lr{Multi-Layer Structural Robustness}) است.
1. **طمع‌ورزی لاجیت در Forward KL:** واگرایی کولبک-لایبلر مستقیم تنها بر روی توزیع خروجی آخرین لایه مدل اعمال می‌شود و به دلیل خاصیت \lr{Zero-Avoiding}، دانش‌آموز را مجبور می‌سازد تا توزیع لاجیت خروجی را به هر قیمتی با معلم تطبیق دهد. این امر موجب نادیده‌گرفتن کامل فضای میانی لایه‌ها و تحریف هندسه زیرفضای پنهان می‌گردد که در شبکه‌های عمیق به افت شدید تعمیم‌پذیری در وظایف چندمرحله‌ای (\lr{Multi-hop Reasoning}) می‌انجامد.
2. **پایداری هندسی با بسته ۶ ستونی:** بسته ۶ ستونی با فرمول‌بندی قضیه ۳.۵ (همگرایی انقباضی خطی با فاکتور انقباض $\kappa^* = \sqrt{1 - \mu^2 / L^2} < 1$) به طور همزمان ۵ زیرفضا را مقید می‌سازد:
   - تراز خطی هسته‌ای لایه‌های پنهان (\lr{Linear CKA} با $\mathcal{L}_{\text{CKA}} \ge 0.92$)
   - قید متعامدی هاوس‌هولدر در آداپتورهای کم‌مرتبه (\lr{HNC Nullspace Continuation})
   - تزریق مرحله‌ای پیوسته (\lr{MOHAWK Homotopy}) جهت مهار تکینگی طیف ویژه
   در نتیجه، هرچند تطابق لاجیت سطحی اندکی تعدیل می‌شود، اما همبستگی چندلایه‌ای بازنمایی حفظ شده و از انحراف مانده در لایه‌های متوالی جلوگیری به عمل می‌آید.

**English Response:**
This reflects the fundamental trade-off between superficial "Logit Greediness" and "Multi-Layer Structural Regularization".
Forward KL strictly minimizes $\mathcal{D}_{\text{KL}}(P_T \parallel P_S)$ solely at the final classifier logits. Because Forward KL is zero-avoiding, it aggressively forces the student to cover all teacher modes at the output layer, completely neglecting intermediate hidden geometry. This leads to internal representation distortion and severe performance collapse on downstream compositional tasks.
In contrast, our 6-Pillar KD Suite (governed by the strict contraction mapping theorem with rate $\kappa^* = \sqrt{1 - \mu^2 / L^2} < 1$, Theorem 3.5 / Eq. 3.42) joint-optimizes logit matching with layer-wise Centered Kernel Alignment ($\mathcal{L}_{\text{CKA}}$), MOHAWK continuous attention homotopy, and Householder nullspace projection. While top-layer cosine similarity is slightly relaxed (0.9184 vs 0.9495), intermediate representation fidelity ($CKA \ge 0.92$) is preserved, guaranteeing robust out-of-distribution generalization.
*References: Master Thesis Section 4.5, Table 4.6 (p. 38), Defense Slides 9-10, Theorem 3.5.*

---

### سوال ۲: سفیدسازی فعال‌سازی‌ها با تجزیه چولسکی و خنثی‌سازی معکوس (Cholesky Whitening & Unwhitening)
**پرسش داور:**
در فشرده‌سازی SVD آگاه از فعال‌سازی، چرا به جای ماتریس وزن، ماتریس سفیدشده $\widetilde{W} = W L$ تجزیه می‌شود و چگونه بدون اعمال مستقیم معکوس چولسکی $L^{-1}$ به داده‌های ورودی، در زمان استنتاج خنثی‌سازی صورت می‌پذیرد؟

**پاسخ به فارسی:**
1. **ضرورت بهینه‌سازی وزنی کوواریانس:** در مدل‌های ترنسفورمر، کانال‌های ورودی دارای واریانس‌های نامتعادل و همبستگی شدید هستند ($\mathbf{H}_X = \mathbb{E}[X X^\top] \ne I$). تجزیه مستقیم SVD بدون در نظر گرفتن $\mathbf{H}_X$ به خطای بازسازی بالا در جهت‌های پرانرژی منجر می‌شود.
2. **سفیدسازی با چولسکی:** چون $\mathbf{H}_X$ مثبت نیمه‌معین و متقارن است، از طریق تجزیه چولسکی به صورت $\mathbf{H}_X = \mathbf{L} \mathbf{L}^\top$ فاکتور می‌شود که $\mathbf{L}$ مثلثی پایینی است. با تبدیل $\widetilde{\mathbf{W}} = \mathbf{W} \mathbf{L}$، ماتریس ورودی موثر سفید شده ($\operatorname{Cov}(\mathbf{L}^{-1} X) = I$) و بنابر قضیه اکرت-یانگ-میرسکی، تجزیه مقادیر منفرد $\widetilde{\mathbf{W}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$ خطای وزن‌دار $\|(W - W_r) \mathbf{H}_X^{1/2}\|_F^2$ را کمینه می‌سازد.
3. **ادغام معکوس در لایه اول پروجکشن:** برای بازیابی خروجی در فاز استنتاج، معکوس چولسکی مستقیماً در فاکتور سمت راست ادغام می‌شود:
   $$W_r^* = \widetilde{\mathbf{W}}_r \mathbf{L}^{-1} = \left(\mathbf{U}_r \mathbf{\Sigma}_r\right) \left(\mathbf{V}_r^\top \mathbf{L}^{-1}\right) = A \cdot B$$
   که در آن $A = \mathbf{U}_r \mathbf{\Sigma}_r \in \mathbb{R}^{d_{\text{out}} \times r}$ و $B = \mathbf{V}_r^\top \mathbf{L}^{-1} \in \mathbb{R}^{r \times d_{\text{in}}}$ هستند. بنابراین در زمان اجرای استنتاج، ماتریس $L^{-1}$ به صورت آفلاین پیش‌محاسبه شده و هیچگونه عملیات معکوس‌سازی یا سربار محاسباتی به گراف زمان اجرا تحمیل نمی‌گردد.

**English Response:**
1. **Activation Covariance Non-Uniformity:** Standard SVD minimizes pure parameter difference $\|W - W_r\|_F^2$, ignoring that activation channels have vastly unequal variances and correlations ($\mathbf{H}_X = \mathbb{E}[XX^\top] \ne I$).
2. **Optimal Whitening via Cholesky:** Because $\mathbf{H}_X \succ 0$, Cholesky decomposition yields $\mathbf{H}_X = \mathbf{L}\mathbf{L}^\top$. Defining the whitened weight $\widetilde{W} = W L$ transforms the objective into $\|(\widetilde{W} - \widetilde{W}_r)\|_F^2$. By the Eckart-Young-Mirsky theorem, the optimal rank-$r$ approximation is given by the truncated SVD of $\widetilde{W} = U_r \Sigma_r V_r^\top$.
3. **Zero-Overhead Static Unwhitening:** The unwhitening operation is statically absorbed into the right low-rank factor prior to deployment:
   $$W_r = U_r \Sigma_r (V_r^\top L^{-1}) = A \cdot B$$
   where $B = V_r^\top L^{-1} \in \mathbb{R}^{r \times d_{\text{in}}}$ is computed once offline. During inference, execution consists of two simple matrix multiplications ($X B^\top A^\top$) with zero matrix inversion latency.
*References: Master Thesis Section 2.4, Eq. 2.18–2.23; Defense Slide 4.*

---

### سوال ۳: قانون پایستگی بودجه بر روی سیمپلکس در WGQA (Simplex Energy Conservation Law)
**پرسش داور:**
علت انفجار پرپلکسیتی مدل GPT-2 در WGQA استاندارد چه بود و چگونه نگاشت سیمپلکس این مشکل را به طور قطعی حل کرد؟

**پاسخ به فارسی:**
1. **علت فیزیکی انفجار پرپلکسیتی:** در WGQA بدون قید، به هر سر کلید-مقدار یک ضریب یادگیرنده بدون قید $w_h \in \mathbb{R}$ تخصیص داده می‌شود که به صورت $\operatorname{scale}_h = \exp(w_h)$ اعمال می‌گردد. در مدل‌های دیکودر علّی ۱۲ لایه‌ای، رانش تصادفی وزن‌ها باعث شد مجموع مقیاس سرها از مقدار تعادلی فاصله گرفته و به میانگین $\alpha > 1$ برسد. این رانش، دمای موثر لاجیت را پیش از اعمال سافت‌مکس به صورت $T_{\text{eff}} = \sqrt{d_k} / \alpha$ فشرده ساخت و در نتیجه، توزیع توجه در طول ۱۲ لایه دچار اشباع قطعی شده و پرپلکسیتی به رقم نجومی **۴,۲۴۸,۶۸۹.۵۰** منفجر گردید.
2. **راهکار پایستگی سیمپلکس:** با تعریف ضریب مقیاس بر روی سیمپلکس نرمال‌شده:
   $$\operatorname{scale}_h = \operatorname{Softmax}(w)_h \times H_{\text{kv}} = \frac{\exp(w_h)}{\sum_{j=1}^{H_{\text{kv}}} \exp(w_j)} \times H_{\text{kv}}$$
   قضیه پایستگی بودجه اثبات می‌کند:
   - مجموع ضرایب همواره دقیقا برابر $H_{\text{kv}}$ باقی می‌ماند: $\sum_{h=1}^{H_{\text{kv}}} \operatorname{scale}_h = H_{\text{kv}}$.
   - با فرض واریانس ورودی $\sigma_K^2$، انرژی مورد انتظار کلیدها ثابت می‌ماند: $\frac{1}{H_{\text{kv}}} \sum \mathbb{E}[\|\tilde{K}_h\|_F^2] = \sigma_K^2$.
   - در نتیجه، دمای موثر لاجیت بدون تغییر باقی مانده و پرپلکسیتی به مقدار پایدار **146.41** مهار گردید.

**English Response:**
In unconstrained WGQA, learnable head scale factors $\operatorname{scale}_h = \exp(w_h)$ induce cumulative scaling drift across deep causal autoregressive layers. If the average head scale exceeds unity ($\alpha > 1$), the effective pre-softmax attention temperature shrinks to $T_{\text{eff}} = \sqrt{d_k}/\alpha$, saturating the attention distribution into one-hot spikes and causing catastrophic perplexity explosion to 4,248,689.50 on GPT-2 WikiText-2.
By projecting the scale vector onto the probability simplex scaled by the number of KV heads ($\operatorname{scale}_h = \operatorname{Softmax}(w)_h \times H_{\text{kv}}$), Theorem 3.4 formally guarantees:
1. Exact budget conservation: $\sum_{h=1}^{H_{\text{kv}}} \operatorname{scale}_h \equiv H_{\text{kv}}$.
2. KV-space energy invariance: $\frac{1}{H_{\text{kv}}} \sum_{h=1}^{H_{\text{kv}}} \mathbb{E}[\|\tilde{K}_h\|_F^2] = \sigma_K^2$.
This eliminates logit temperature drift entirely, stabilizing causal autoregressive perplexity to 146.41.
*References: Master Thesis Section 4.6, Theorem 3.4, Table 4.7; Defense Slide 6.*

---

### سوال ۴: فرمول‌بندی مسئله بهینه‌سازی HNC (HNC Optimization Problem & Regularization)
**پرسش داور:**
روش HNC-SVD چگونه مسئله بهینه‌سازی را تعریف می‌کند و نقش تقریب گاوس-نیوتن هسین در مهار خطای انباشتی چیست؟

**پاسخ به فارسی:**
1. **فرمول‌بندی مسئله منظم‌شده تیخونوف:** در فشرده‌سازی عمیق مدل‌هایی نظیر RoBERTa (۱۲ لایه)، برش مستقیم مقادیر منفرد در لایه‌های متوالی خطای خطی را انباشته کرده و دقت را به ۴۷.۸۲٪ فرومی‌پاشد. مسئله بهینه‌سازی HNC گام جابجایی وزن $\Delta \theta$ را تحت ماتریس هسین/فیشر $G = J^\top J \succeq 0$ به صورت زیر تعریف می‌کند:
   $$\min_{\Delta \theta} \left\{ \langle \nabla \phi(\theta_0), \Delta \theta \rangle + \frac{1}{2}\|\Delta \theta\|_2^2 + \frac{1}{2\mu} \Delta \theta^\top G \Delta \theta \right\}$$
2. **حل دقیق و انطباق با فضای پوچ:** با مشتق‌گیری درجه اول، جواب تحلیلی و یکتای این مسئله به دست می‌آید:
   $$\Delta \theta^* = -\left( I + \frac{1}{\mu} G \right)^{-1} \nabla \phi_{\text{SVD}}(\theta_0)$$
   در راستاهایی که مقادیر ویژه هسین به صفر میل می‌کنند (فضای پوچ $G v \approx 0$)، ضریب میرایی $(1 + \lambda/\mu)^{-1} \to 1$ بوده و گام فشرده‌سازی با آزادی کامل اعمال می‌شود. اما در راستاهای با انحنای شدید هسین ($G v \gg 0$)، گام فشرده‌سازی به شدت میرا شده و مانع از تخریب خروجی لایه‌ها می‌گردد. این روش دقت فشرده‌سازی ۴ لایه RoBERTa را به **71.67\%** (+23.85% بهبود) ارتقا می‌دهد.

**English Response:**
Direct layer-by-layer low-rank truncation causes cumulative error compounding in deep networks (e.g., RoBERTa accuracy collapsing to 47.82% under 4-layer $r=128$ SVD).
Householder Nullspace Continuation (HNC) formulates weight perturbation as a Tikhonov-regularized quadratic minimization over the Gauss-Newton curvature manifold $G = J^\top J \succeq 0$:
$$\min_{\Delta \theta} \left\{ \langle \nabla \phi(\theta_0), \Delta \theta \rangle + \frac{1}{2}\|\Delta \theta\|_2^2 + \frac{1}{2\mu} \Delta \theta^\top G \Delta \theta \right\}$$
The closed-form stationary solution is:
$$\Delta \theta^* = -\left(I + \frac{1}{\mu} G\right)^{-1} \nabla \phi_{\text{SVD}}(\theta_0)$$
Along directions spanning the nullspace of Fisher/Hessian ($G v \approx 0$), perturbation occurs unhindered. Along high-curvature directions ($G v \gg 0$), displacement is damped by $(1 + \lambda_i/\mu)^{-1}$, preserving functional representation with $\mathcal{O}(\epsilon^2)$ error and lifting RoBERTa 4-layer compressed accuracy to 71.67% (+23.85% margin).
*References: Master Thesis Section 3.3, Theorem 3.3, Eq. 3.28; Defense Booklet p. 5.*

---

### سوال ۵: تفاوت حساسیت SVD در SST-2 در برابر MNLI (SVD on SST-2 vs MNLI)
**پرسش داور:**
چرا فشرده‌سازی SVD بدون کالیبراسیون در وظیفه SST-2 عملکرد نسبتا قابل قبولی نشان می‌دهد، اما در وظیفه MNLI با افت چشمگیر مواجه می‌شود؟

**پاسخ به فارسی:**
1. **تفاوت پیچیدگی معنایی و ابعاد زیرفضا:** مجموعه داده SST-2 یک وظیفه دسته‌بندی احساسی تک‌جمله‌ای بر روی دو کلاس است. در این وظیفه، قطبیت کلمات (نظیر "good" یا "terrible") حتی با برهم‌نهی زیرفضاهای کم‌بعد و رتبه بسیار پایین ($r \le 32$) قابل تفکیک باقی می‌ماند.
2. **استنتاج رابطه‌ای چندجمله‌ای در MNLI:** وظیفه MNLI شامل جفت‌جملات (فرض و حکم) در ۳ کلاس (موافق، متناقض، خنثی) است. استنتاج منطقی نیازمند نگاشت دقیق دوطرفه توجه متقابل و تقاطع فضاهای ویژگی است. در فاکتورسازی SVD کالیبره‌نشده، رتبه ماتریس توجه دوخطی به $\min(T, r)$ محدود شده و بعد فضای پوچ به $\operatorname{dim}(\operatorname{null}(M)) \ge d - r$ افزایش می‌یابد (قضیه ۳.۱ رساله). این امر موجب ناورایی امتیاز توجه نسبت به تفاوت‌های ظریف نحوی شده و دقت به **34.10\%** (هم‌سطح حدس تصادفی در ۳ کلاس) فرومی‌پاشد. تنها با سفیدسازی چولسکی و HNC است که دقت به **72.20\%** احیا می‌گردد.

**English Response:**
1. **Single-Sentence Semantic Separation (SST-2):** SST-2 is binary single-sentence sentiment classification where sentiment polarity is linearly separable even after drastic singular value truncation ($r \le 32$).
2. **Multi-Sentence Cross-Attention Geometry (MNLI):** MNLI requires fine-grained 3-way logical inference across premise-hypothesis pairs. By Theorem 3.1, naive low-rank projection bounds the bilinear attention matching operator rank to $\operatorname{rank}(M) \le r$, expanding the nullspace to $\operatorname{dim}(\operatorname{null}(M)) \ge d - r$. Because $S(x_j, x_k + \Delta x) = S(x_j, x_k)$ for any nullspace perturbation, subtle cross-sentence relational cues are obliterated, collapsing uncalibrated accuracy to 34.10% (pure random choice on 3-class classification). Calibration via Cholesky whitening and ResSVD restores this to 72.20%.
*References: Master Thesis Chapter 4, Table 4.4; Defense Booklet p. 6.*

---

### سوال ۶: پروتکل‌های ارزیابی و استقلال بنچ‌مارک (Baseline Protocols)
**پرسش داور:**
پروتکل شما برای سنجش عادلانه میان ۱۳۱ مکانیزم توجه و ۷۲ فشرده‌ساز چه بوده است تا از تداخل حافظه، اثرات گرم‌کردن و بهینه‌سازی‌های خاص جلوگیری شود؟

**پاسخ به فارسی:**
تمامی ۲۰۳ ماژول بر روی یک سخت‌افزار واحد فیزیکی (NVIDIA RTX 4070 Laptop GPU, 8GB GDDR6, SM 8.9) تحت پروتکل کنترل‌شده زیر ارزیابی شدند:
- **هم‌شرطی ابعاد تانسور:** ورودی‌ها با طول توالی ثابت ($T=128$)، اندازه دسته ($B=16$) و بعد پنهان مدل ($D=768$) تغذیه شدند.
- **تضمین گرم‌کردن سخت‌افزار:** پیش از هر اندازه‌گیری، ۲۵ گام گرم‌کردن (\lr{Warm-up}) برای پایدارسازی فرکانس کلاک و حافظه نهان GPU اعمال گردید.
- **سنکرون‌سازی دقیق رخدادهای گرافیکی:** تمامی زمان‌سنجی‌ها با شیءهای بومی \lr{\texttt{torch.cuda.Event(enable\_timing=True)}} و با اعمال \lr{\texttt{torch.cuda.synchronize()}} انجام گرفت تا تاخیر ناهمگام CPU در محاسبات مداخله نکند.
- **ایزولاسیون حافظه:** بعد از اجرای هر ماژول، حافظه نهان PyTorch با \lr{\texttt{torch.cuda.empty\_cache()}} تخلیه شد و حداکثر حافظه تخصیص‌یافته توسط \lr{\texttt{max\_memory\_allocated()}} ثبت گردید.

**English Response:**
All 203 modules were benchmarked on a strictly controlled physical testbench (NVIDIA RTX 4070 GPU, SM 8.9):
- Fixed tensor geometries ($B=16, T=128, D=768$) across all runs.
- 25 dedicated warm-up iterations to reach steady-state GPU clocks and cache warmness.
- High-precision timing via asynchronous native `torch.cuda.Event(enable_timing=True)` coupled with full device barriers (`torch.cuda.synchronize()`).
- Strict memory isolation with garbage collection and `torch.cuda.empty_cache()` between trials, capturing peak VRAM via `torch.cuda.max_memory_allocated()`.
*References: Master Thesis Section 4.1; RESULTS_PROVENANCE.md.*

---

### سوال ۷: مبنای ریاضی آستانه ۳ سیگما (3\sigma Quality Gate Criterion)
**پرسش داور:**
آستانه 56.00% برای قبولی در دروازه کیفیت (\lr{Quality Gate PASS}) بر چه مبنای آماری پایه‌گذاری شده است؟

**پاسخ به فارسی:**
این آستانه بر مبنای آزمون فرضیه آماری دو جمله‌ای بر روی مجموعه داده متوازن SST-2 با $N = 872$ نمونه با شانس حدس تصادفی $p_0 = 0.50$ استخراج شده است:
$$\sigma = \sqrt{\frac{p_0 (1 - p_0)}{N}} = \sqrt{\frac{0.5 \times 0.5}{872}} \approx 0.01693 \quad (1.693\%)$$
کران بحرانی ۳ انحراف معیار ($3\sigma$) برای رد قاطع حدس تصادفی با سطح اطمینان ۹۹.۸۷٪ عبارت است از:
$$p_{\text{crit}} = p_0 + 3\sigma = 0.50 + 3 \times 0.01693 = 0.5508 \quad (55.08\%)$$
برای اعمال حاشیه اطمینان مهندسی، این مقدار به سمت بالا گرد شده و روی **56.00\%** تثبیت گردیده است تا اطمینان حاصل شود که هیچ مدلی بر اساس بایاس تصادفی کلاس یا حدس غیرمعنادار نشان قبولی دریافت نمی‌کند.

**English Response:**
Derived from a two-tailed binomial hypothesis test against random guessing ($H_0: p_0 = 0.50$) on the balanced SST-2 validation split ($N=872$):
$$\sigma = \sqrt{\frac{p_0(1-p_0)}{N}} = \sqrt{\frac{0.25}{872}} \approx 0.01693$$
The $3\sigma$ threshold for rejecting the null hypothesis with 99.87% statistical confidence yields:
$$p_{\text{crit}} = 0.50 + 3(0.01693) = 0.5508 \quad (55.08\%)$$
With conservative engineering headroom, the quality gate threshold was established at **56.00%**, strictly discarding models operating on empirical majority priors.
*References: Master Thesis Section 3.6, Eq. 3.48–3.50.*

---

### سوال ۸: تزریق مدل‌های فضای حالت علّی (Mamba) در معماری‌های دوطرفه (Causal SSM on Masked Models)
**پرسش داور:**
مدل‌های فضای حالت مانند Mamba ماهیتا علّی (یک‌طرفه) هستند. چگونه آن‌ها را در مدل‌های دوطرفه نظیر RoBERTa یا DistilBERT تزریق کردید بدون آنکه فرآیند شکست بخورد؟

**پاسخ به فارسی:**
در تزریق مدل‌های SSM در انکودرهای دوطرفه، دو نوآوری فنی پیاده‌سازی شد:
1. **اسکن تکه‌ای دوطرفه (\lr{Bidirectional Chunked Recurrence}):** توالی ورودی در دو جهت پیشرو ($1 \to T$) و پسرو ($T \to 1$) توسط بلوک‌های فضای حالت اسکن شده و خروجی‌ها با پروجکشن خطی فیوز می‌گردند.
2. **پدینگ هوشمند و ماسک توجه انکودر:** مدیریت بردار حالت در توکن‌های پد شده با ایزوله‌سازی حالت مخفی $h_t$ انجام گرفت تا توکن‌های فیلر باعث آلودگی بردار حالت توکن‌های معنادار نشوند.

**English Response:**
To inject inherently unidirectional causal state-space models into bidirectional masked encoders:
1. **Bidirectional Chunked Scanning:** Forward and time-reversed sequences are processed through paired recurrent kernels with linear output blending.
2. **Mask-Aware State Reset:** Recurrent carryover tensors are zero-masked at padding positions, preventing dummy padding tokens from polluting cumulative hidden states.
*References: Master Thesis Chapter 3; scripts/test_all_causality.py.*

---

### سوال ۹: پدیده تورم پارامتر در ۸۰ ماژول فشرده‌سازی (The Parameter Inflation Paradox)
**پرسش داور:**
در تحلیل عیب‌یابی ماژول‌های ناکارآمد، پدیده «تورم پارامتر» (\lr{Parameter Inflation}) چیست و چرا رخ می‌دهد؟

**پاسخ به فارسی:**
در تجزیه کم‌مرتبه ماتریس $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ با رتبه $r$، دو ماتریس $A \in \mathbb{R}^{d_{\text{out}} \times r}$ و $B \in \mathbb{R}^{r \times d_{\text{in}}}$ جایگزین می‌شوند. تعداد پارامترهای جدید برابر $r(d_{\text{in}} + d_{\text{out}})$ است.
«رتبه بحرانی» که در آن تعداد پارامترها بدون تغییر می‌ماند عبارت است از:
$$r_{\text{crit}} = \frac{d_{\text{in}} d_{\text{out}}}{d_{\text{in}} + d_{\text{out}}}$$
برای لایه استاندارد $768 \times 768$، مقدار $r_{\text{crit}} = 384$ است. در صورتی که طراح رتبه را بدون توجه به این کران بزرگتر از ۳۸۴ (مثلا $r=512$) انتخاب کند، یا آداپتورهای کم‌مرتبه بدون ادغام در کنار وزن پایه نگه داشته شوند، نه تنها فشرده‌سازی رخ نمی‌دهد، بلکه حجم مدل تا ۱.۲۵ برابر متورم می‌شود. راهکار فریم‌ورک ما اعمال سقف بودجه قطعی $r \le 0.45 \cdot r_{\text{crit}}$ و ادغام کامل وزن‌ها در فاز صادرات است.

**English Response:**
Replacing $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ with $A \in \mathbb{R}^{d_{\text{out}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{in}}}$ yields $r(d_{\text{in}} + d_{\text{out}})$ parameters. The breakeven critical rank is:
$$r_{\text{crit}} = \frac{d_{\text{in}} d_{\text{out}}}{d_{\text{in}} + d_{\text{out}}}$$
For a $768 \times 768$ projection, $r_{\text{crit}} = 384$. Any selected rank $r > 384$ (e.g., $r=512$) or unmerged residual adapter inadvertently expands total parameter storage up to $1.25\times$. Our framework enforces an automated sanity budget clamp: $r \le \lfloor 0.45 \cdot r_{\text{crit}} \rfloor$ with compulsory offline matrix fusion.
*References: UNDERPERFORMING_AND_BUGGY_MODULES_FORENSIC_ANALYSIS.md Pattern F.*

---

### سوال ۱۰: تحلیل مدل روف‌لاین و شدت محاسباتی کرنل‌ها (Roofline Model & Arithmetic Intensity)
**پرسش داور:**
شدت محاسباتی (\lr{Arithmetic Intensity}) کرنل‌های تریتون شما در نمودار روف‌لاین چگونه ارزیابی شده است؟

**پاسخ به فارسی:**
شدت محاسباتی به صورت نسبت کل عملیات ممیز شناور بر حسب بایت‌های تبادل‌شده از حافظه اصلی تعریف می‌شود:
$$\mathcal{I} = \frac{\text{FLOPs}}{\text{Bytes Transferred from DRAM}}$$
- در مکانیزم توجه استاندارد Dense SDPA، به دلیل بارگذاری و ذخیره مکرر ماتریس امتیاز $N \times N$، شدت محاسباتی بسیار پایین است ($\mathcal{I} < 15 \text{ FLOP/B}$) و کرنل کاملاً در ناحیه محدود به حافظه (\lr{Memory-Bound}) قرار می‌گیرد.
- در کرنل‌های فیوزشده تریتون ما (نظیر Linear Attn و GDN Scan)، با کاشی‌بندی در SRAM و اسکن تکه‌ای، مادی‌سازی ماتریس $N \times N$ حذف شده و شدت محاسباتی به $75 \text{ to } 98 \text{ FLOP/B}$ ارتقا یافته است که اجرای محاسبات را به سقف پهنای باند و مرز محاسباتی نزدیک می‌سازد.

**English Response:**
Operational arithmetic intensity is formulated as $\mathcal{I} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$.
Standard dense attention materializes the full $N \times N$ score matrix in global HBM, resulting in low arithmetic intensity ($\mathcal{I} < 15 \text{ FLOP/B}$) and severe memory-bus saturation.
Our custom Triton JIT kernels (Linear Attention, GDN Scan, HGRN2) maintain state in on-chip SRAM registers via chunked associative scans, eliminating HBM round-trips and lifting intensity to $75\text{--}98 \text{ FLOP/B}$, achieving up to 95% of peak hardware memory bandwidth on physical GPU benchmarks.
*References: docs/COMPARISON.md Section 6; Master Thesis Section 4.7.*

---

### سوال ۱۱: پشته گسیل سه‌سطحی و تضمین عدم بروز خطا (3-Level Dispatch & Zero-Failure Fallbacks)
**پرسش داور:**
پشته گسیل سه‌سطحی پلتفرم NeuralOps چگونه کار می‌کند و چه مزیتی در محیط‌های ناهمگن دارد؟

**پاسخ به فارسی:**
سیستم گسیل هوشمند در `neuralops/core/cuda_kernels/` عملگرها را در ۳ سطح اجرا می‌کند:
1. **سطح ۱ (Native CUDA C++):** ۱۸ کرنل فوق بهینه‌شده با کامپایلر nvcc، بهره‌گیری از دستورات سطح وارپ و ثبات‌ها (تا ۱۳۳۶ برابر شتاب).
2. **سطح ۲ (OpenAI Triton JIT):** ۲۶ کرنل کامپایل درجا برای سرعت و قابلیت حمل میان پردازنده‌های مدرن.
3. **سطح ۳ (Pure PyTorch Reference):** پیاده‌سازی برداری استاندارد ریاضی که در صورت عدم پشتیبانی GPU یا بروز ناسازگاری به صورت کامپیوتر-محور فعال شده و برابری عددی دقیق را بدون کرش کردن تضمین می‌نماید.
همچنین میانبر CPU Fast-Path در صورت دریافت تانسور روی رم سیستم، بلافاصله سطح ۳ را اجرا می‌کند تا زمان صرف ارزیابی کرنل گرافیکی نگردد.

**English Response:**
The 3-level execution hierarchy provides zero-failure reliability across heterogeneous compute targets:
- **Level 1 (Native CUDA C++):** 18 bare-metal kernels optimized with warp shuffles and shared memory tiling (delivering up to $1336.96\times$ speedup on MoE routing).
- **Level 2 (OpenAI Triton JIT):** 26 JIT-compiled kernels balancing architectural portability with high SRAM occupancy.
- **Level 3 (Pure PyTorch):** Vectorized reference implementations ensuring bit-level mathematical parity as an unconditional fallback.
A CPU fast-path bypasses GPU driver queries when inputs reside in host RAM, avoiding scheduling latency.
*References: Master Thesis Section 4.7, Table 4.8; Defense Slides 16-17.*

---

### سوال ۱۲: انقباض نگاشت تبدیل هاوس‌هولدر در Gated DeltaNet (Householder Contraction Proof)
**پرسش داور:**
چرا در قضیه ۳.۴ ادعا شده است که نگاشت حافظه در Gated DeltaNet یک انقباض اکید است؟

**پاسخ به فارسی:**
معادله به‌روزرسانی حافظه در Gated DeltaNet عبارت است از:
$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$$
ماتریس $H_t = I - \beta_t k_t k_t^\top$ با انتخاب $\beta_t = \frac{2}{\|k_t\|_2^2}$ به یک عملگر بازتاب هاوس‌هولدر تبدیل می‌شود که متعامد و متقارن است ($\|H_t\|_2 = 1$).
هنگامی که گیت میرایی $\alpha_t \in (0, 1)$ اعمال می‌گردد:
$$\|S_t\|_F \le \alpha_t \|S_{t-1}\|_F \|H_t\|_2 + \|v_t k_t^\top\|_F = \alpha_t \|S_{t-1}\|_F + \|v_t k_t^\top\|_F$$
تحت شرط تحریک پیوسته $\frac{1}{\tau}\sum_{s=t}^{t+\tau-1} \frac{k_s k_s^\top}{\|k_s\|_2^2} \succeq \alpha I$، مقدار ویژه بزرگتر ماتریس تبدیل اکیداً کوچکتر از ۱ شده و خطای حافظه به صورت هندسی منقبض می‌شود: $\mathbb{E}[\|E_t\|_F^2] \le \rho^t \|E_0\|_F^2$ با $\rho < 1$. این امر پایداری عددی اسکن در توالی‌های بی‌نهایت بلند را تضمین می‌سازد.

**English Response:**
In Gated DeltaNet recurrence $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$, setting $\beta_t = \frac{2}{\|k_t\|_2^2}$ yields an exact Householder reflection matrix with $\|H_t\|_2 = 1$.
Under gating $\alpha_t \in (0, 1)$ and the persistent excitation condition $\frac{1}{\tau}\sum_{s=t}^{t+\tau-1}\frac{k_s k_s^\top}{\|k_s\|_2^2} \succeq \alpha I$, the cumulative state transition operator becomes a strict geometric contraction:
$$\mathbb{E}[\|E_t\|_F^2] \le \rho^t \|E_0\|_F^2, \quad \rho < 1$$
This bounded-input bounded-state (BIBS) property formally prevents gradient explosion during associative scanning.
*References: Master Thesis Section 3.4, Theorem 3.4; Defense Booklet p. 5.*

---

### سوال ۱۳: استریمینگ بافر KV در ASVD و صرفه‌جویی ۸ برابری حافظه (ASVD KV Cache Streaming)
**پرسش داور:**
مکانیزم ASVD چگونه توانسته بدون افت دقت پرپلکسیتی، حجم بافر KV را ۸ برابر کاهش دهد؟

**پاسخ به فارسی:**
در تولید خودبازگشتی، ذخیره تانسورهای کامل کلید و مقدار برای تمام سرهای توجه به ازای طول توالی‌های بلند حجم عظیمی از VRAM را اشغال می‌کند. مکانیزم ASVD پروجکشن‌های $W_k$ و $W_v$ را به یک فضای نهان فشرده مشترک $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll H d_h$) نگاشت می‌دهد.
در زمان استنتاج، تنها بردار نهان فشرده در بافر حافظه ذخیره می‌شود که حجم بافر را از ۲۴ گیگابایت در طول ۶۴K به زیر ۳ گیگابایت ($8\times$ کاهش) می‌رساند. بازسازی مقادیر اصلی به صورت برخط و ادغام‌شده در رجیسترهای پرسرعت GPU انجام می‌شود.

**English Response:**
Autoregressive decoding typically requires storing full multi-head Key-Value activations in HBM. ASVD projects keys and values into a shared compressed low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ where $d_c \ll H d_h$.
Only the compact latent vector is cached across decoding steps, reducing 64K context KV footprint from $>24$ GB down to $<3$ GB ($8\times$ memory reduction), while on-the-fly reconstruction is executed directly in fast SRAM registers.
*References: Master Thesis Chapter 2; Defense Slide 7.*

---

### سوال ۱۴: کران تحلیلی خطای مانتیس در کوانتایزاسیون FP8 E4M3 (FP8 Quantization Mantissa Bound)
**پرسش داور:**
چرا فرمت E4M3 در کوانتایزاسیون ماتریس‌های منفرد برگزیده شد و کران خطای آن چیست؟

**پاسخ به فارسی:**
فرمت FP8 E4M3 دارای ۴ بیت توان و ۳ بیت مانتیس است. خطای نسبی ناشی از قطع مانتیس در این فرمت دارای کران تحلیلی زیر است:
$$\frac{|\hat{x} - x|}{|x|} \le 2^{-(m+1)} = 2^{-4} = 0.0625 \quad (6.25\%)$$
در آزمایش‌های عملی ما روی ضرایب مقادیر منفرد $\Sigma$， با اعمال مقیاس‌بندی پویا (\lr{Dynamic Scaling})، خطای نسبی تجربی به **0.0264** (۲.۶۴٪) رسید که بسیار پایین‌تر از سقف مجاز است. همچنین حفظ ۳ بیت مانتیس دقت تفکیک‌پذیری مقادیر ویژه بزرگ را تضمین کرده و افت پرپلکسیتی را به کمتر از ۰.۱۵ واحد محدود می‌سازد.

**English Response:**
The FP8 E4M3 format (4 exponent bits, 3 mantissa bits) provides a formal theoretical mantissa truncation error bound of:
$$\frac{|\hat{x} - x|}{|x|} \le 2^{-(m+1)} = 2^{-4} = 0.0625 \quad (6.25\%)$$
Coupled with block-wise dynamic scaling, empirical relative error drops to $0.0264$ (2.64%), well below the 5% error ceiling, perfectly preserving top singular values and restricting perplexity degradation below 0.15 points.
*References: Master Thesis Chapter 2; Defense Slide 6.*

---

### سوال ۱۵: صحت گرادیان و ارزیابی چنددانه‌ای (Multi-Seed Cross-Validation & Gradcheck)
**پرسش داور:**
آیا صحت پیاده‌سازی کرنل‌های CUDA و Triton از نظر محاسبات گرادیان و پایداری در دانه‌های تصادفی مختلف بررسی شده است؟

**پاسخ به فارسی:**
بله، در فرآیند راستی‌آزمایی:
- تمام ۲۰ کرنل اختصاصی تحت آزمون سخت‌گیرانه \lr{\texttt{torch.autograd.gradcheck}} با خطای تحلیلی زیر $10^{-5}$ قرار گرفتند و ۱۰۰ تکرار انتشار پیشرو و پسرو را بدون رانش عددی ثبت کردند.
- ارزیابی چنددانه‌ای با دانه‌های تصادفی ۴۲، ۱۳۳۷ و ۲۰۲۶ بر روی تمام ۲۰۳ ماژول اجرا گردید و انحراف معیار دقت در تمام ماژول‌های پایدار زیر ۰.۳۵٪ به دست آمد که تکرارپذیری قطعی نتایج را اثبات می‌نماید.

**English Response:**
Full gradient validity and stochastic reproducibility are certified:
- All 20 native CUDA/Triton kernels passed rigorous `torch.autograd.gradcheck` with analytical-numerical error under $10^{-5}$ across 100 consecutive forward-backward iterations.
- Randomized multi-seed cross-validation (seeds 42, 1337, 2026) demonstrated tight accuracy standard deviation ($\sigma < 0.35\%$), confirming robust empirical convergence.
*References: docs/reports/CROSS_VALIDATION_REGRESSION_REPORT.md; multi_seed_stress_test_131_72.py.*

---

### سوال ۱۶: چرخش موضعی درجای امبدینگ مکانی (Fused RoPE in-place Transformation)
**پرسش داور:**
کرنل اختصاصی `fused_rope.py` چه بهینه‌سازی سخت‌افزاری را محقق ساخته است؟

**پاسخ به فارسی:**
پیاده‌سازی متداول RoPE در پایتورچ نیازمند برش بردارها، اعمال توابع مثلثاتی سینوس و کسینوس و پیوند تانسوری مجدد است که باعث تخصیص مکرر بافرهای میانی در VRAM و کاهش پهنای باند حافظه می‌شود.
کرنل تریتون اختصاصی ما چرخش دوبعدی جفت مولفه‌ها ($[-x_2, x_1]$) را به صورت درجا در ثبات‌های SRAM بدون نیاز به حافظه موقت انجام داده و شتاب **2.05 برابری** به ثبت رسانده است.

**English Response:**
Standard PyTorch RoPE slices tensors, generates trigonometric matrices, and re-concatenates outputs, causing redundant HBM allocations. Our custom `fused_rope.py` kernel performs 2D Givens coordinate rotation in-place directly within SRAM registers, eliminating transient memory buffers and achieving a $2.05\times$ wall-clock speedup.
*References: Master Thesis Section 4.7; Table 4.8.*

---

### سوال ۱۷: تنظیم خودکار کرنل‌ها با Triton Autotuner (Hardware Auto-Tuning)
**پرسش داور:**
تنظیم پارامترهای اجرایی کرنل‌های تریتون (مانند اندازه بلوک و تعداد وارپ) چگونه انجام پذیرفته است؟

**پاسخ به فارسی:**
تمامی کرنل‌های پرکاربرد به متدولوژی \lr{\texttt{triton.autotune}} مجهز هستند که در اولین اجرا، فضای پیکربندی شامل اندازه‌های بلوک ($B_M, B_N \in \{32, 64, 128\}$)، تعداد وارپ‌ها ($num\_warps \in \{2, 4, 8\}$) و مراحل خط لوله ($num\_stages \in \{2, 3, 4\}$) را ارزیابی کرده و بهینه‌ترین پیکربندی را با توجه به معماری پردازنده (Ada Lovelace SM 8.9) ذخیره می‌نماید.

**English Response:**
Kernels leverage `triton.autotune` sweeps evaluating tile dimensions ($BLOCK_M, BLOCK_N \in \{32, 64, 128\}$), warp allocations ($num\_warps \in \{2, 4, 8\}$), and prefetch pipelining stages ($num\_stages \in \{2, 3, 4\}$), dynamically selecting optimal configurations matched to the physical SM 8.9 cache topology.
*References: neuralops/triton_kernels/autotuner.py.*

---

### سوال ۱۸: هرس خبرگان در توزیع بلند با REAP-MoE (Expert Pruning in Long-Tail Distributions)
**پرسش داور:**
در شکاف پنجم مرز دانش، مکانیزم REAP-MoE چگونه بدون آسیب به تخصص‌های نادر، خبرگان شبکه را هرس می‌کند؟

**پاسخ به فارسی:**
هرس سنتی خبرگان بر مبنای بسامد فراخوانی عمل می‌کند که موجب حذف فاجعه‌بار خبرگان مرتبط با کارهای نادر (دم بلند توزیع) می‌شود.
روش پیشنهادی REAP-MoE اهمیت خبرگان را بر مبنای ضرب نُرم گرادیان در واریانس بازنمایی فعال‌سازی ارزیابی می‌کند. در آزمایش‌های ما بر روی شبکه با ۴ خبره، کاهش خبرگان به ۲ خبره با حفظ ۱۰۰٪ دقت بازیابی سوزن در انبار کاه (\lr{Needle In A Haystack}) به انجام رسید.

**English Response:**
Standard MoE pruning relies solely on call frequency, obliterating rare-domain specialist experts residing in the long tail. REAP-MoE computes an activation-gradient Riemannian metric that isolates informational density. Pruning from 4 to 2 experts retained 100% retrieval fidelity on the 32K Needle-In-A-Haystack benchmark.
*References: Master Thesis Chapter 2; Defense Slide 6.*

---

### سوال ۱۹: مقیاس‌پذیری در طول توالی تا ۸K و مهار پیچیدگی حافظه (Sequence Scaling to 8192)
**پرسش داور:**
رفتار مدل‌های پیشنهادی در مقایسه با توجه متراکم استاندارد در طول توالی‌های بالا چگونه است؟

**پاسخ به فارسی:**
همانطور که در نمودار اسلاید ۱۵ و بخش ۴.۴ رساله مستند شده است:
- در توجه متراکم استاندارد (Dense Softmax)، تاخیر زمانی با شیب درجه دوم $\mathcal{O}(N^2)$ رشد کرده و در طول توالی $N=8192$ با خطای سرریز حافظه (\lr{CUDA OOM}) مواجه می‌شود.
- در مکانیزم‌های خطی بهینه‌شده (نظیر Cosformer و Mamba)، پیچیدگی حافظه و زمان به صورت کاملاً خطی $\mathcal{O}(N)$ رفتار کرده و تاخیر در طول توالی ۸۱۹۲ توکن در زیر ۱۰ میلی‌ثانیه مهار می‌گردد.

**English Response:**
As proven in Slide 15 and Section 4.4, Dense Softmax exhibits quadratic $\mathcal{O}(N^2)$ scaling, culminating in CUDA OOM at $N=8192$. Our hardware-fused linear and state-space architectures maintain strict $\mathcal{O}(N)$ scaling, keeping inference latency under 10 ms at $N=8192$.
*References: Master Thesis Section 4.4; Defense Slide 15.*

---

### سوال ۲۰: بازیافت خطای مانده ResSVD (Residual Error Recovery Mechanics)
**پرسش داور:**
ماژول ResSVD چگونه خطای برش مقادیر منفرد را بازیافت می‌کند؟

**پاسخ به فارسی:**
پس از فاکتورسازی رتبه-$r$ ماتریس وزن $W \approx W_r = U_r \Sigma_r V_r^\top$، ماتریس خطای مانده $E = W - W_r$ محاسبه می‌شود. ماژول ResSVD یک نگاشت تنک یا زیرفضای کم‌مرتبه فرعی ($r_{\text{res}} \ll r$) برای مهار موثرترین جهات این خطای مانده اختصاص می‌دهد:
$$W_{\text{final}} = W_r + U_{\text{res}} \Sigma_{\text{res}} V_{\text{res}}^\top$$
این مرحله در جدول ابلیشن ۵ مرحله‌ای، دقت مدل فشرده را از 71.50% به **72.20%** افزایش داده و مانع از اشباع خطا در لایه‌های انتهایی می‌شود.

**English Response:**
Following rank-$r$ truncation $W \approx W_r = U_r \Sigma_r V_r^\top$, the residual matrix $E = W - W_r$ is formed. ResSVD allocates an auxiliary low-rank pathway ($r_{\text{res}} \ll r$) targeted at the dominant singular modes of the residual manifold:
$$W_{\text{final}} = W_r + U_{\text{res}} \Sigma_{\text{res}} V_{\text{res}}^\top$$
In the 5-stage ablation pipeline, this lifts final model accuracy from 71.50% to **72.20%**, stabilizing representation fidelity across deep stacks.
*References: Master Thesis Chapter 3; Defense Booklet p. 6; Defense Slide 8b.*

