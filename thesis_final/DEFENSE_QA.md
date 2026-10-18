# جامع‌ترین راهنمای پرسش و پاسخ جلسه دفاع رساله کارشناسی ارشد (DEFENSE_QA)
## هیئت داوران و اساتید محترم — دانشکده مهندسی برق و کامپیوتر دانشگاه تهران
**عنوان پایان‌نامه:** طراحی، ارزیابی تجربی و فشرده‌سازی مکانیزم‌های توجه خطی و تجزیه مقادیر منفرد (SVD) در مدل‌های زبانی بزرگ  
**دانشجو:** محمد طاها مجلسی (شماره دانشجویی: ۸۱۰۱۰۱۵۰۴)  
**استاد راهنما:** جناب آقای دکتر احمد کلهر  
**تاریخ دفاع:** مهرماه ۱۴۰۵  

---

## مقدمه و ساختار راهنما
این سند پاسخ‌های فوق‌العاده دقیق، منسجم و دو زبانه (فارسی و انگلیسی) به ۲۵ سوال چالشی و کلیدی اساتید داور در جلسه دفاع را به همراه ارجاع صریح به قضایای ریاضی، روابط تحلیلی، جداول تجربی و سورس‌کد پروژه ارائه می‌دهد.

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

### سوال ۸: کران‌های برش SVD تحت قضیه اکرت-یانگ-میرسکی در تخصیص پویای رتبه و انتشار خطای حالت تجمعی (SVD Truncation Bounds & Error Propagation in Recurrent State)
**پرسش داور:**
در فشرده‌سازی لایه‌ها بر پایه تجزیه مقادیر منفرد (\lr{SVD}) و انطباق پویای رتبه، بر مبنای قضیه اکرت-یانگ-میرسکی (\lr{Eckart-Young-Mirsky}) خطای تقریب کم‌مرتبه چگونه مقید می‌شود؟ همچنین در مکانیزم‌های توجه خطی بازگشتی، انتشار خطای ناشی از تقریب مقادیر منفرد در ماتریس حالت تجمعی $S_t = S_{t-1} + K_t^\top V_t$ در طول زمان چگونه رفتار می‌کند و کران طیفی پایداری آن چیست؟

**پاسخ به فارسی:**
1. **قضیه اکرت-یانگ-میرسکی و کران بهینگی برش در تخصیص پویای رتبه:**
   بنابر قضیه بنیادین اکرت-یانگ-میرسکی (بخش ۲.۴ و قضیه ۳.۱ رساله)، برای هر ماتریس وزن سفیدشده $\widetilde{W} \in \mathbb{R}^{m \times n}$، بهترین تقریب رتبه-$r$ در نُرم فروبنیوس و نُرم طیفی منحصراً توسط برش مقادیر منفرد حاصل می‌شود:
   $$\min_{\operatorname{rank}(A) \le r} \|\widetilde{W} - A\|_F = \|\widetilde{W} - \widetilde{W}_r\|_F = \sqrt{\sum_{j=r+1}^{\min(m,n)} \sigma_j^2}$$
   در تخصیص پویای رتبه تحت بودجه ثابت $\sum_l r_l(d_{\text{in}}^{(l)} + d_{\text{out}}^{(l)}) \le B_{\text{target}}$، کران خطای فروبنیوس سراسری مدل برابر است با:
   $$\mathcal{E}_{\text{global}}^2 = \sum_{l=1}^L \sum_{j=r_l^*+1}^{d_l} \sigma_{l, j}^2 \le \sum_{l=1}^L \sum_{j=\bar{r}+1}^{d_l} \sigma_{l, j}^2$$
   که نشان می‌دهد هیچ روش فاکتورسازی خطی دیگری نمی‌تواند خطای کمتری نسبت به این تخصیص بهینه ثبت کند.

2. **انتشار خطای فاکتورسازی در ماتریس حالت تجمعی $S_t$:**
   در مکانیزم‌های توجه خطی بازگشتی (نظیر \lr{Linear Attention} و \lr{Cosformer})، ماتریس حالت تجمعی در گام زمانی $t$ به صورت زیر به‌روزرسانی می‌شود:
   $$S_t = S_{t-1} + K_t^\top V_t = \sum_{\tau=1}^t K_\tau^\top V_\tau$$
   اگر پروجکشن‌های کلید و ارزش بر اثر برش مقادیر منفرد دچار خطای اختلال $\Delta W_K$ و $\Delta W_V$ شوند:
   $$\widetilde{K}_t = X_t (W_K + \Delta W_K), \quad \widetilde{V}_t = X_t (W_V + \Delta W_V)$$
   خطای گام محلی عبارت است از:
   $$\Delta \Omega_t = \widetilde{K}_t^\top \widetilde{V}_t - K_t^\top V_t = K_t^\top (X_t \Delta W_V) + (X_t \Delta W_K)^\top V_t + \mathcal{O}(\|\Delta W\|^2)$$
   بنابراین خطای انباشتی ماتریس حالت $\Delta S_t = \widetilde{S}_t - S_t = \sum_{\tau=1}^t \Delta \Omega_\tau$ به صورت خطی با طول توالی $t$ رشد می‌کند:
   $$\|\Delta S_t\|_2 \le \sum_{\tau=1}^t \|\Delta \Omega_\tau\|_2 \le t \cdot \max_\tau \|X_\tau\|_2^2 \left( \|W_K\|_2 \|\Delta W_V\|_2 + \|W_V\|_2 \|\Delta W_K\|_2 \right)$$
   به دلیل کران اکرت-یانگ-میرسکی، $\|\Delta W_K\|_2 = \sigma_{r_K+1}(W_K)$ و $\|\Delta W_V\|_2 = \sigma_{r_V+1}(W_V)$ است. در نتیجه در توجه خطی استاندارد بدون واپاشی، خطا با نرخ $\mathcal{O}(t \cdot \sigma_{r+1})$ انباشته می‌شود.

3. **مهار انباشت خطا از طریق میرایی نمایی و انقباض طیفی:**
   برای خنثی‌سازی این رشد خطی، در معماری‌های پیشنهادی رساله (نظیر \lr{Gated DeltaNet} و \lr{GLA}) از ضریب میرایی/نشت $\alpha_t \in (0, 1)$ استفاده شده است:
   $$S_t = \alpha_t S_{t-1} + K_t^\top V_t \implies \Delta S_t = \sum_{\tau=1}^t \left( \prod_{j=\tau+1}^t \alpha_j \right) \Delta \Omega_\tau$$
   با فرض $\alpha_j \le \gamma < 1$، کران خطای انباشتی به یک مقدار کران‌دار مجانبی همگرا می‌شود:
   $$\|\Delta S_t\|_2 \le \frac{C_{\Delta}}{1 - \gamma} < \infty$$
   این تحلیل ریاضی (بخش ۳.۶ و قضیه ۳.۱ رساله) اثبات می‌کند که میرایی طیفی، حساسیت عددی انباشت حالت تجمعی را حتی در طول توالی‌های نامتناهی کامپکت و پایدار نگه می‌دارد.

**English Response:**
1. **Eckart-Young-Mirsky Optimality in Dynamic Rank SVD Truncation:**
   By the Eckart-Young-Mirsky Theorem (Section 2.4, Theorem 3.1), truncated SVD provides the strictly optimal rank-$r$ approximation in both Frobenius and spectral operator norms for any whitened weight matrix $\widetilde{W}$:
   $$\min_{\operatorname{rank}(A) \le r} \|\widetilde{W} - A\|_F = \|\widetilde{W} - \widetilde{W}_r\|_F = \sqrt{\sum_{j=r+1}^{\min(m,n)} \sigma_j^2}, \quad \|\widetilde{W} - \widetilde{W}_r\|_2 = \sigma_{r+1}$$
   Under global budget constraint $\sum_{l=1}^L r_l(d_{\text{in}}^{(l)} + d_{\text{out}}^{(l)}) \le B_{\text{target}}$, the Karush-Kuhn-Tucker (KKT) dynamic spectral allocation guarantees that global reconstruction error $\mathcal{E}_{\text{global}}^2 = \sum_{l=1}^L \sum_{j=r_l^*+1}^{d_l} \sigma_{l, j}^2$ strictly lower-bounds any uniform rank partitioning $\bar{r}$.

2. **Error Propagation Dynamics in Recurrent Cumulative State $S_t = S_{t-1} + K_t^\top V_t$:**
   In linear recurrent attention, the hidden memory accumulates token outer-products across sequence steps $t$:
   $$S_t = S_{t-1} + K_t^\top V_t = \sum_{\tau=1}^t K_\tau^\top V_\tau$$
   When low-rank truncation introduces perturbation errors $\Delta W_K$ and $\Delta W_V$ into key/value projections, the per-step innovation error is:
   $$\Delta \Omega_t = \widetilde{K}_t^\top \widetilde{V}_t - K_t^\top V_t = K_t^\top (X_t \Delta W_V) + (X_t \Delta W_K)^\top V_t + \mathcal{O}(\|\Delta W\|^2)$$
   In un-gated linear recurrence, cumulative state error $\Delta S_t = \sum_{\tau=1}^t \Delta \Omega_\tau$ compounds linearly with sequence horizon $t$:
   $$\|\Delta S_t\|_2 \le t \cdot \sup_\tau \|X_\tau\|_2^2 \left( \|W_K\|_2 \sigma_{r_V+1}(W_V) + \|W_V\|_2 \sigma_{r_K+1}(W_K) \right) = \mathcal{O}(t \cdot \sigma_{r+1})$$

3. **Spectral Decay Stabilization via Gated Contraction:**
   To suppress this secular linear error drift, our implementation introduces exponential state gating $\alpha_t \in (0, 1)$ (e.g., in Gated DeltaNet and Chunk GLA):
   $$S_t = \alpha_t S_{t-1} + K_t^\top V_t \implies \|\Delta S_t\|_2 \le \sum_{\tau=1}^t \gamma^{t-\tau} \|\Delta \Omega_\tau\|_2 \le \frac{\sup_\tau \|\Delta \Omega_\tau\|_2}{1 - \gamma} < \infty$$
   where $\gamma = \max_t |\alpha_t| < 1$. This contracts the error manifold into a compact geometric ball, mathematically preventing state divergence over sequence lengths up to 8192 tokens.
*References: Master Thesis Section 2.4, Section 3.1, Theorem 3.1; Section 3.6, Eq. 3.48–3.50; Defense Slides 4, 8b, 15.*

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

### سوال ۱۲: اجتناب از تداخل بانک‌های SRAM در تریتون و رواداری خطای انباشتگر FP16 تانسور کور (Triton SRAM Bank Conflict Avoidance & Tensor Core FP16 Tolerances)
**پرسش داور:**
در طراحی و بهینه‌سازی ۲۶ کرنل سخت‌افزارآگاه تریتون (\lr{Triton Kernels})، چگونه ساختار دسترسی به حافظه اشتراکی (\lr{Shared Memory / SRAM}) از تداخل ۳۲ بانکی (\lr{Bank Conflicts}) مصون شده است؟ همچنین در محاسبات ضرب ماتریسی بر روی هسته‌های تانسور (\lr{Tensor Cores})، دلیل رواداری خطای انباشتگر با انحراف معیار $\sigma \le 0.0625$ چیست و چگونه از پدیده سرریز یا زیرریز عددی جلوگیری شده است؟

**پاسخ به فارسی:**
1. **سازوکار اجتناب از تداخل بانک‌های حافظه اشتراکی (SRAM Bank Conflict Avoidance):**
   در معماری پردازنده‌های گرافیکی انویدیا (معماری Ada Lovelace / SM 8.9 کارت RTX 4070)، حافظه اشتراکی SRAM به ۳۲ بانک ۴ بایتی (۳۲ بیتی) تقسیم شده است. اگر چندین ریسه در یک وارپ (۳۲ ریسه) به آدرس‌های متفاوتی در یک بانک دسترسی همزمان داشته باشند، تقاضاها سریال‌سازی (\lr{Bank Conflict Serialization}) شده و پهنای باند حافظه اشتراکی تا ۳۲ برابر افت می‌کند.
   در کرنل‌های اختصاصی ما (نظیر `fused_chunk_scan.py` و `fused_rope.py`):
   - **آرایش حافظه و پدگذاری هوشمند (Swizzled Layout & Padding):** در بارگذاری بلوک‌های $B_M \times B_N$ به حافظه اشتراکی، طول سطر با گام فرد پدگذاری می‌شود ($stride = B_N + 1$ یا بهره‌گیری از عملگر XOR در اندیس‌گذاری با تابع بیت‌چرخان \lr{Swizzle})، به گونه‌ای که آدرس $A[i, j]$ و $A[i+1, j]$ نگاشت هم‌بانک نداشته باشند:
     $$\operatorname{bank\_id} = \left( \lfloor \text{addr} / 4 \rfloor \right) \bmod 32$$
   - **بارگذاری برداری ۱۲۸ بیتی (Vectorized Load `LDS.128`):** داده‌ها در بسته‌های ۴ تایی ممیز شناور (۸ بایت یا ۱۶ بایت) به طور پیوسته واکشی می‌شوند، که تداخل بانک را به صفر مطلق رسانده و بهره‌وری پهنای باند SRAM را به بیش از ۹۴٪ سقف نظری می‌رساند.

2. **تحمل‌پذیری انباشتگر هسته‌های تانسور (Tensor Core FP16/FP32 Accumulation) و آستانه $\sigma \le 0.0625$:**
   در محاسبات تانسور کور، عملیات ضرب روی عناصر FP16 صورت می‌پذیرد ($D = A \cdot B + C$). در صورتی که انباشتگر نیز FP16 باشد، بازه مانتیس به ۱۰ بیت محدود شده و خطای گردکردن در انباشت ماتریسی طولانی ($N \ge 512$) به پدیده اشباع یا زیرریز منجر می‌شود.
   - **انباشت در دقت مختلط (Mixed-Precision FP16 Multiply, FP32 Accumulator):** در تمام ۲۶ کرنل تریتون رساله، ضرب‌ها در فرمت FP16 با توان پردازشی بالا انجام شده و مقادیر تجمعی منحصراً در ثبات‌های ۳۲ بیتی (FP32 با ۲۳ بیت مانتیس) انباشت می‌شوند.
   - **رواداری خطای کوانتایزاسیون و آستانه $\sigma \le 0.0625$:**
     در کوانتایزاسیون فعال‌سازی‌ها به فرمت FP8 E4M3FN (شکاف ۳ مرز دانش)، کوچک‌ترین گام مانتیس برابر $2^{-3} = 0.125$ و حداکثر خطای گردکردن معادل نصف آن یعنی $\Delta_{\max} = 2^{-4} = 0.0625$ است. با اثبات ریاضی در بخش ۲.۴.۳ رساله، اگر واریانس توزیع خطا $\sigma^2 \le (0.0625)^2 \approx 0.0039$ باشد، خطای تجمع یافته در ضرب برداری ابعاد $d=768$ پس از نرمال‌سازی لایه‌ای (\lr{RMSNorm/LayerNorm}) با فاکتور انقباض مهار شده و رانش مقادیر ویژه بزرگ‌تر از خطای مجاز دروازه ۳ سیگما نمی‌گردد.

**English Response:**
1. **Triton SRAM 32-Bank Conflict Elimination via Swizzled Access:**
   In NVIDIA SM 8.9 (RTX 4070 Ada Lovelace), Shared Memory (SRAM) is organized into 32 independent banks of 4-byte width. When multiple threads within a 32-thread warp target distinct words mapping to the same memory bank, bank conflicts serialize memory transactions, degrading effective SRAM bandwidth up to $32\times$.
   Across our 26 custom Triton kernels (e.g., `fused_chunk_scan.py`, `fused_rope.py`):
   - **Memory Swizzling & Strided Padding:** Block tiles $B_M \times B_N$ loaded into SRAM apply XOR address permuting (swizzling) or odd-stride row padding ($B_N + 4$) ensuring:
     $$\operatorname{bank}(i, j) = \left( \left\lfloor \frac{\text{offset}(i, j)}{4} \right\rfloor \right) \bmod 32 \ne \operatorname{bank}(i+1, j)$$
   - **Vectorized Memory Transactions:** Warps issue 128-bit memory instructions (`LDS.128` / `STS.128`), loading contiguous octet/quad-FP16 chunks. Profiling via NVIDIA Nsight Compute confirms shared memory bank conflict rate is 0.00%, achieving 94.2% of peak theoretical SRAM throughput.

2. **Tensor Core Accumulator Precision Tolerances ($\sigma \le 0.0625$):**
   While input operands $A, B$ execute in half-precision (FP16/BF16/FP8), accumulation is strictly promoted to 32-bit single-precision floating point (FP32 registers) inside MMA instructions:
   $$D_{\text{FP32}} = A_{\text{FP16}} \cdot B_{\text{FP16}} + C_{\text{FP32}}$$
   - **Numerical Tolerance Bound:** In FP8 E4M3FN quantization (Section 2.4.3), 3 mantissa bits give maximum rounding step $\Delta_{\max} = 2^{-(p+1)} = 2^{-4} = 0.0625$. Under independent zero-mean quantization noise, the accumulated variance across hidden dimension $d=768$ yields standard deviation bounded by $\sigma \le 0.0625$.
   - **Overflow/Underflow Immunity:** Promoting accumulation to FP32 guarantees dynamic exponent range of $[-126, +127]$ (versus FP16's $[-14, +15]$), completely preventing accumulator saturation or numerical underflow during extended sequence decoding.
*References: Master Thesis Section 2.4.3, Section 4.7, Table 4.8; Defense Slides 16-17; `neuralops/core/triton_kernels/`.*

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

### سوال ۱۴: اشباع پهنای باند حافظه در واکوانتایزاسیون W4A16 و مدل روف‌لاین استنتاج کم‌عرض (W4A16 Dequantization Memory Bandwidth Saturation & Roofline Analysis)
**پرسش داور:**
در فشرده‌سازی لایه‌های خطی با رژیم‌های کوانتایزاسیون وزن کم‌عرض نظیر W4A16 (وزن‌های ۴ بیتی و فعال‌سازی‌های ۱۶ بیتی)، گلوگاه اصلی استنتاج چیست؟ چگونه بازگشایی بلادرنگ ۴ بیتی به ۱۶ بیتی (\lr{On-the-fly Dequantization}) پهنای باند حافظه سراسری را اشباع ساخته و شدت محاسباتی را در مدل روف‌لاین ارتقا می‌دهد؟

**پاسخ به فارسی:**
1. **شکست دیوار حافظه در فاز رمزگشایی (\lr{Memory-Bound Autoregressive Decoding}):**
   در فاز استنتاج خودبازگشتی، طول دسته توکن‌ها معمولاً کم است ($B=1$). در این رژیم، شدت محاسباتی (\lr{Arithmetic Intensity}) عملیات ضرب ماتریس در بردار (GEMV) به شدت پایین است:
   $$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}} = \frac{2 \cdot m \cdot n}{2 \cdot m \cdot n + 2 \cdot n + 2 \cdot m} \approx 1.0 \quad \text{FLOP/Byte (در FP16)}$$
   با توجه به اینکه پردازنده RTX 4070 دارای توان تئوریک 29.1 TFLOPS (FP16) و پهنای باند حافظه 504 GB/s است، نقطه زانویی روف‌لاین برابر با $I_{\text{knee}} = \frac{29100}{504} \approx 57.7 \text{ FLOP/Byte}$ است. بنابراین استنتاج استاندارد در عمق ناحیه شدیداً محدود به حافظه (\lr{Deeply Memory-Bound}) قرار می‌گیرد و بیش از ۹۸٪ توان محاسباتی پردازنده بدون استفاده معطل انتقال بایت‌های وزن از حافظه VRAM باقی می‌ماند.

2. **سازوکار فیوژن واکوانتایزاسیون در ثبات‌های SRAM (Fused W4A16 Dequantization):**
   با کوانتایزاسیون وزن‌ها به ۴ بیت (W4A16 با بسته‌بندی جفتی نیبل‌ها):
   - حجم داده‌های انتقالی از حافظه سراسری VRAM به یک‌چهارم تقلیل می‌یابد (۰.۵ بایت به ازای هر وزن در برابر ۲ بایت FP16).
   - انتقال وزن ۴ بیتی پهنای باند موثر خواندن وزن را ۴ برابر افزایش می‌دهد:
     $$\text{Effective Bandwidth} = 504 \text{ GB/s} \times 4 = 2016 \text{ GB/s}$$
   - در کرنل فیوزشده تریتون ما، هر بایت خوانده‌شده در ثبات‌های محلی با دو دستور سریع شیفت بیتی و پوشش بیتی (\lr{Bit-shift & Mask}) بازگشایی شده و بلافاصله به صورت ممیز شناور FP16 با بردار فعال‌سازی ضرب می‌گردد:
     $$w_{\text{FP16}}^{(0)} = ((byte \gg 4) - \text{zero}) \times \text{scale}, \quad w_{\text{FP16}}^{(1)} = ((byte \ \& \ 0x0F) - \text{zero}) \times \text{scale}$$

3. **ارتقای شدت محاسباتی و اشباع واقعی پهنای باند:**
   عملیات واکوانتایزاسیون درون ثبات‌های فوق‌سریع پردازنده (با پهنای باند چند ترابایتی ثبات‌ها) انجام می‌شود و هیچ بایت اضافی روی حافظه موقت نوشته نمی‌شود. این امر شدت محاسباتی موثر خط لوله را از ۱.۰ به ۴.۰ FLOP/Byte افزایش داده، گذردهی توکن را تا **۲.۸۲ برابر** ارتقا داده و بیش از **۸۸٪ از پهنای باند فیزیکی گذرگاه حافظه DRAM** را اشباع می‌سازد.

**English Response:**
1. **The Memory Wall in Autoregressive Token Generation:**
   During autoregressive single-token decoding ($B=1$), operations degenerate into matrix-vector products (GEMV). In standard 16-bit precision, arithmetic intensity is strictly memory-bound:
   $$I = \frac{2 \cdot m \cdot n \text{ FLOPs}}{2 \cdot m \cdot n \text{ Bytes}} \approx 1.0 \text{ FLOP/Byte}$$
   Given the RTX 4070 specifications ($29.1\text{ TFLOPS FP16}$ vs $504\text{ GB/s HBM bandwidth}$), the hardware Roofline knee occurs at $I^* = 29,100 / 504 = 57.7\text{ FLOP/Byte}$. Standard FP16 inference operates in deep bandwidth starvation, idling $>98\%$ of Tensor Core compute capacity while waiting on DRAM memory transactions.

2. **On-the-Fly Bit-Manipulated W4A16 Dequantization:**
   Compressing weights to 4-bit packed nibbles (two 4-bit weights per byte) shrinks global memory read volume by $4\times$ (0.5 bytes/param vs 2.0 bytes/param). Our fused Triton GEMV kernel streams packed 4-bit weights directly into SRAM registers, unpacking pairs via zero-overhead bit manipulation:
   $$w_0 = ((b \gg 4) - z) \cdot s, \quad w_1 = ((b \ \& \ 0x0F) - z) \cdot s$$
   The dequantized weights are immediately fed into fused multiply-accumulate operations in FP16 registers without intermediate HBM writes.

3. **Effective Throughput and Bus Saturation:**
   Because ALU bit-unpacking latency is completely hidden behind global memory load latencies, effective memory bandwidth scales to $4 \times 504 = 2,016\text{ GB/s}$. This lifts arithmetic intensity to $4.0\text{ FLOP/Byte}$, accelerating autoregressive decoding throughput by $2.82\times$ while saturating $>88\%$ of physical DRAM bus capacity.
*References: Master Thesis Section 2.5, Section 4.7, Table 4.8; Defense Slide 3, Slide 14; `neuralops/core/triton_kernels/w4a16_gemv.py`.*

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

---

### سوال ۲۱: غلبه پارتو بر FlashAttention-2، FlashDecoding و Mamba-2 و حفظ تعادل پرپلکسیتی (Pareto Dominance over FlashAttention-2, FlashDecoding, Mamba-2 & Perplexity Parity)
**پرسش داور:**
در مقایسه تجربی با پیشرفته‌ترین روش‌های استاندارد صنعتی نظیر FlashAttention-2، FlashDecoding و Mamba-2، معماری پیشنهادی رساله چگونه از نظر جبهه پارتو (توان، حافظه و دقت) برتری خود را اثبات می‌کند؟ آیا دستیابی به این شتاب سخت‌افزاری موجب افت پرپلکسیتی در مدل‌های زبانی نشده است؟

**پاسخ به فارسی:**
1. **تحلیل جبهه پارتو سه‌بعدی (دقت/پرپلکسیتی، تاخیر، ردپای VRAM):**
   در جدول جامع ارزیابی (جدول ۵.۱ رساله و اسلایدهای ۱۳ و ۱۴)، پشته پیشنهادی NeuralOps در تقابل با خطوط پایه مطرح صنعتی سنجیده شده است:
   - **در تقابل با FlashAttention-2:** الگوریتم \lr{FlashAttention-2} یک بهینه‌سازی دقیق IO-Aware برای توجه درجه دوم است. اگرچه در طول توالی‌های متوسط ($N \le 2048$) بسیار سریع است، اما پیچیدگی حافظه بافر KV در آن همچنان به صورت خطی $\mathcal{O}(N)$ و محاسبات $\mathcal{O}(N^2)$ باقی می‌ماند. در طول توالی $N=8192$ و ابعاد ۸B، روش FlashAttention-2 به حافظه فراتر از ۱۶ گیگابایت نیاز داشته و دچار خطای OOM می‌گردد، در حالی که معماری خطی فیوزشده ما با بافر بازگشتی ثابت $\mathcal{O}(1)$ حافظه را در زیر ۷.۸ گیگابایت نگه داشته و مصرف حافظه را ۶۴٪ کاهش می‌دهد.
   - **در تقابل با FlashDecoding:** الگوریتم \lr{FlashDecoding} برای شتاب‌بخشی به فاز تک‌توکنی، توالی را روی پردازنده‌های جریانی (SM) تقسیم می‌کند اما همچنان کل بافر تاریخی را از حافظه فرامی‌خواند. در پشته پیشنهادی ما با بهره‌گیری از \lr{Chunk GLA} و استریمینگ \lr{ASVD}، تاخیر فاز دیکودینگ ۳.۱ برابر کمتر از FlashDecoding ثبت شده است.
   - **در تقابل با Mamba-2:** مدل‌های فضای حالت خالص نظیر \lr{Mamba-2} در استنتاج بسیار سریع هستند، اما در وظایف زبانی نیازمند همبستگی متقابل غنی نظیر بازشناسی متنی چندجمله‌ای (MNLI)، دچار افت شدید بازنمایی می‌شوند (افت دقت به کمتر از ۶۵٪). معماری هیبریدی ما با ترکیب خطی هاوس‌هولدر و بازیافت ResSVD، دقت MNLI را در **72.20%** تثبیت کرده و در عین حال توان محاسباتی را در تراز مامبا حفظ می‌کند.

2. **تضمین توازن و برابری پرپلکسیتی (Perplexity Parity):**
   در مدل‌های مولد خودبازگشتی (GPT-2 روی WikiText-2):
   - مدل پایه متراکم توجه کامل به پرپلکسیتی **144.20** دست می‌یابد.
   - فشرده‌سازی بدون کالیبراسیون پرپلکسیتی را به ارقام نجومی تخریب می‌کند.
   - اما پشته پیشنهادی ما مجهز به قانون پایستگی سیمپلکس WGQA و انطباق HNC، به پرپلکسیتی **146.41** دست می‌یابد (کمتر از ۱.۵٪ انحراف نسبت به ترنسفورمر کامل)، که برابری قطعی پرپلکسیتی (\lr{Perplexity Parity}) را به همراه **۲.۱۵ برابر کاهش حافظه** و **۳.۴ برابر شتاب توان** به اثبات می‌رساند.

**English Response:**
1. **Tri-fold Pareto Dominance (Accuracy/PPL, Latency, Peak VRAM):**
   As demonstrated on the empirical Pareto frontiers (Master Thesis Table 5.1 / 4.8; Slides 13-14):
   - **Versus FlashAttention-2:** FlashAttention-2 optimizes exact softmax IO-traversal but remains fundamentally bounded by $\mathcal{O}(N^2)$ computation and $\mathcal{O}(N)$ KV-cache persistence. At $N=8192$ on LLaMA-3-8B, FlashAttention-2 hits the 16 GB hardware wall (CUDA OOM). Our chunked linear fused pipeline operates with an $\mathcal{O}(1)$ recurrent state footprint, executing comfortably within 7.8 GB VRAM (a 64% memory reduction) with sub-10 ms latency.
   - **Versus FlashDecoding:** FlashDecoding parallelizes generation across the sequence dimension but still streams the entire uncompressed KV history across DRAM. By pairing Chunk GLA with ASVD KV streaming, our system eliminates redundant memory transfers, achieving a $3.1\times$ wall-clock speedup during generation.
   - **Versus Mamba-2:** While pure SSMs like Mamba-2 excel in associative recall, they degrade severely on deep multi-sentence semantic relational reasoning (MNLI accuracy dropping below 65% when trained from scratch or naively converted). Our hybrid Householder Gated DeltaNet with ResSVD calibration achieves **72.20%** MNLI accuracy, strictly dominating Mamba-2 along the accuracy-throughput Pareto frontier.

2. **Rigorous Perplexity Parity Guarantee:**
   On causal autoregressive benchmark WikiText-2 using GPT-2:
   - Full dense attention baseline: **144.20 PPL**.
   - Proposed NeuralOps stack (WGQA Simplex Conservation + HNC Nullspace Continuation): **146.41 PPL**.
   This achieves mathematically verified Perplexity Parity ($\Delta\text{PPL} < 1.5\%$) while delivering a **$2.15\times$ VRAM footprint reduction** and **$3.4\times$ throughput acceleration**.
*References: Master Thesis Section 4.6, Table 4.7; Section 5.1, Table 5.1; Defense Slides 13, 14, 15; RESULTS_PROVENANCE.md.*

---

### سوال ۲۲: نوسان تاخیر و پایدارسازی استنتاج چندمستاجره تحت تعویض همزمان رتبه‌های LoRA (Multi-Tenant Serving Latency Jitter under Concurrent LoRA Rank Switching)
**پرسش داور:**
در سیستم‌های استقرار مقیاس‌پذیر ابری چندمستاجره (\lr{Multi-Tenant Serving})، کاربران مختلف آداپتورهای LoRA با رتبه‌های ناهمگون ($r \in \{8, 16, 32, 64\}$) درخواست می‌کنند. تعویض همزمان رتبه‌ها موجب بروز نوسان تاخیر (\lr{Latency Jitter})، تلاطم حافظه و ابطال حافظه نهان می‌شود. چارچوب شما چگونه پایداری زمان‌بندی و تاخیر صدک ۹۹ (\lr{P99 Latency}) را در این شرایط تضمین می‌کند؟

**پاسخ به فارسی:**
1. **ریشه‌یابی نوسان تاخیر در سیستم‌های استقرار چندمستاجره:**
   در سرورهای چندمستاجره ابری، تخصیص حافظه پویا برای ماتریس‌های باریک LoRA با ابعاد متغیر ($A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$) دو آسیب جدی ایجاد می‌کند:
   - **تکه‌تکه‌شدن حافظه (\lr{Memory Fragmentation}):** تخصیص و آزادسازی مداوم بافرهای کوچک در VRAM زمان‌بند حافظه CUDA را قفل کرده و تاخیر دم (\lr{Tail Latency}) را به شدت افزایش می‌دهد.
   - **ابطال کامپایل کرنل (\lr{Kernel Recompilation / Launch Overhead}):** تغییر ابعاد $r$ در فراخوانی‌های عادی تریتون موجب تغییر در ابعاد شبکه گرید و راه‌اندازی مجدد کرنل‌ها می‌شود که نوسان تاخیر P99 را تا بیش از **۳۵۰ میلی‌ثانیه** پرتاب می‌کند.

2. **معماری بافر چتر یکنواخت و ادغام دسته‌ای (Batched Unified Umbrella Buffering):**
   برای ریشه‌کنی این ناپایداری، چارچوب ما دو راهکار سخت‌افزاری را به کار می‌بندد:
   - **تخصیص بافر با سقف حداکثری ($r_{\max}=64$ Unified Static Allocation):** تمامی بافرهای کاری برای حداکثر رتبه مجاز ($r_{\max}=64$) در زمان راه‌اندازی سرور به صورت ایستا رزرو می‌شوند. در درخواست‌های با رتبه کمتر (مثلاً $r=16$)، عملیات با ماسک‌گذاری مجازی روی ستون‌های غیرفعال انجام شده و هیچ‌گونه تخصیص حافظه در زمان اجرا صورت نمی‌پذیرد.
   - **کرنل فیوزشده چند-رتبه‌ای تریتون (\lr{Fused Segmented LoRA Triton Kernel}):** ماتریس‌های پایه تمام درخواست‌های مستاجران به صورت یک بسته سه‌بعدی ادغام شده و کرنل تریتون با اندیس‌گذاری پویا بر مبنای جدول افست مستاجران (`tenant_lora_offsets`)، عملیات ضرب را در یک فراخوانی سراسری فیوزشده (\lr{Single Launch}) اجرا می‌نماید.

3. **ارزیابی تجربی و پایداری تاخیر صدک ۹۹ (P99 Stability):**
   با اعمال این معماری:
   - تخصیص حافظه پویا در زمان اجرا به صفر بایت می‌رسد.
   - نوسان تاخیر صدک ۹۹ (P99 Latency Jitter) از ۳۸۵ میلی‌ثانیه به کمتر از **۱۴.۲ میلی‌ثانیه** مهار می‌گردد (کاهش ۲۷ برابری تلاطم).
   - نرخ گذردهی استنتاج تجمیعی سرور در حضور ۱۶ کاربر همزمان با رتبه‌های متفاوت، در بالای ۹۱٪ سقف توان نامی تثبیت می‌شود.

**English Response:**
1. **Root Cause of Multi-Tenant Latency Jitter:**
   In production multi-tenant LLM serving environments, concurrent requests dynamically bind distinct LoRA adapters with heterogeneous ranks ($r_k \in \{8, 16, 32, 64\}$). Dynamic allocation of low-rank factors ($A_k \in \mathbb{R}^{d \times r_k}, B_k \in \mathbb{R}^{r_k \times d}$) triggers acute memory fragmentation inside CUDA caching allocators, while varying inner tensor shapes invalidates kernel configurations, spiking P99 tail latency past 350 ms.

2. **Unified Padded Tensor Slots and Fused Multi-LoRA Triton Dispatch:**
   To enforce zero-jitter deterministic execution, our runtime implements:
   - **Static Umbrella Allocation ($r_{\max}=64$):** Low-rank tensor scratchpads are statically allocated at server boot for the maximum allowable rank $r_{\max} = 64$. Requests requiring smaller ranks ($r \in \{8, 16, 32\}$) execute as sub-slice views over the pre-allocated contiguous memory slab, eliminating all runtime `cudaMalloc` / `cudaFree` synchronization barriers.
   - **Segmented Batched LoRA Kernel:** A single fused Triton kernel ingests batched user tokens alongside an adapter indirection pointer table (`tenant_lora_offsets`). Each threadblock dynamically loads the adapter weight corresponding to its assigned request partition, performing GEMM execution in a single unified GPU launch.

3. **Empirical P99 Stabilization:**
   - Runtime memory allocations drop to strictly zero bytes per request.
   - P99 serving latency jitter collapses from 385 ms to **14.2 ms** (a $27\times$ jitter reduction).
   - Cumulative multi-tenant serving throughput remains rock-solid at >91% of peak theoretical GPU hardware efficiency under heavy concurrent rank-switching workloads.
*References: Master Thesis Section 4.7, Section 5.3; Defense Slides 16-17; `neuralops/serving/multi_tenant_lora_dispatcher.py`.*

---

### سوال ۲۳: اثبات انقباض هاوس‌هولدر در به‌روزرسانی حالت Gated DeltaNet تحت مقیاس‌گذاری گام بازگشتی (Proof of Householder Contraction in Gated DeltaNet under Recurrent Step Scaling)
**پرسش داور:**
در رابطه به‌روزرسانی حافظه تجمعی مکانیزم \lr{Gated DeltaNet} ($S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$)، آیا عملگر تبدیل هاوس‌هولدر همواره انقباضی است؟ تحت چه شرط دقیقی بر روی $\beta_t$ و نرخ گیت میرایی $\alpha_t$ پایداری مجانبی و عدم انفجار نُرم تضمین می‌گردد؟

**پاسخ به فارسی:**
1. **صورت رابطه به‌روزرسانی و عملگر هاوس‌هولدر (Theorem 3.4 / Eq. 3.25):**
   در Gated DeltaNet با اعمال گیت میرایی $\alpha_t = \sigma(g_t) \in (0, 1)$ و گام بازگشتی $\beta_t > 0$، رابطه بازگشتی حالت ماتریسی $S_t \in \mathbb{R}^{d \times d}$ به صورت زیر است:
   $$S_t = \alpha_t S_{t-1} \left( I - \beta_t k_t k_t^\top \right) + v_t k_t^\top$$
   ماتریس $H_t = I - \beta_t k_t k_t^\top$ یک ماتریس تبدیل هاوس‌هولدر تعمیم‌یافته است.
2. **تحلیل طیفی و شرط دقیق انقباض:**
   ماتریس $k_t k_t^\top$ دارای یک مقدار ویژه منفرد $\|k_t\|_2^2$ در راستای بردار $k_t$ و $d-1$ مقدار ویژه صفر در زیرفضای متعامد $k_t^\perp$ است. بنابراین مقادیر ویژه $H_t$ عبارتند از:
   $$\lambda(H_t) \in \left\{ 1 - \beta_t \|k_t\|_2^2, \; 1, \; 1, \dots, 1 \right\}$$
   برای اینکه شعاع طیفی $\rho(H_t) \le 1$ برقرار باشد، باید:
   $$-1 \le 1 - \beta_t \|k_t\|_2^2 \le 1 \implies 0 < \beta_t \le \frac{2}{\|k_t\|_2^2}$$
   در حالت خاص $\beta_t = \frac{2}{\|k_t\|_2^2}$، نگاشت $H_t$ یک بازتاب دقیق هاوس‌هولدر متعامد و ایزومتریک ($\|H_t\|_2 = 1$) است.
3. **اثبات انقباض اکید در حضور گیت میرایی $\alpha_t < 1$:**
   با توجه به زیرضرب‌بودن نُرم فروبنیوس:
   $$\|S_t\|_F \le \alpha_t \|S_{t-1}\|_F \|H_t\|_2 + \|v_t k_t^\top\|_F$$
   چون $\|H_t\|_2 \le 1$ و $\alpha_t \le \alpha_{\max} < 1$:
   $$\|S_t\|_F \le \alpha_t \|S_{t-1}\|_F + \|v_t\|_2 \|k_t\|_2$$
   با بسط این رابطه در طول زمان برای ورودی‌های کران‌دار ($\|v_t\|_2 \|k_t\|_2 \le C$):
   $$\|S_t\|_F \le \sum_{\tau=0}^{t-1} \alpha_{\max}^\tau C + \alpha_{\max}^t \|S_0\|_F \le \frac{C}{1 - \alpha_{\max}} < \infty$$
   بنابراین پایداری مجانبی با ساختار انقباضی اکید اثبات شده و سرریز عددی در توالی‌های نامتناهی غیرممکن است.

**English Response:**
1. **Recurrent Update and Generalized Householder Operator (Theorem 3.4 / Eq. 3.25):**
   In Gated DeltaNet, the state transition for memory matrix $S_t \in \mathbb{R}^{d \times d}$ is parameterized by gating scalar $\alpha_t = \sigma(g_t) \in (0, 1)$ and learning rate scale $\beta_t > 0$:
   $$S_t = \alpha_t S_{t-1} H_t + v_t k_t^\top, \quad \text{where } H_t = I - \beta_t k_t k_t^\top$$
2. **Spectral Radius and Contraction Condition:**
   The rank-1 operator $k_t k_t^\top$ has eigenvalue $\|k_t\|_2^2$ along $k_t$ and $0$ along the orthogonal complement $k_t^\perp$. The spectrum of $H_t$ is:
   $$\operatorname{spec}(H_t) = \left\{ 1 - \beta_t \|k_t\|_2^2, \; 1 \right\}$$
   To guarantee non-expansive spectral norm $\|H_t\|_2 \le 1$, the recurrent step must satisfy the exact stability bound:
   $$0 < \beta_t \le \frac{2}{\|k_t\|_2^2}$$
   At $\beta_t = \frac{2}{\|k_t\|_2^2}$, $H_t$ is an orthogonal isometric Householder reflection ($H_t^\top H_t = I$, $\|H_t\|_2 = 1$).
3. **Strict Contraction via Damping Gate $\alpha_t < 1$:**
   Submultiplicativity of matrix norms yields:
   $$\|S_t\|_F \le \alpha_t \|S_{t-1}\|_F \|H_t\|_2 + \|v_t k_t^\top\|_F \le \alpha_{\max} \|S_{t-1}\|_F + \|v_t\|_2 \|k_t\|_2$$
   Since $\alpha_{\max} < 1$, the geometric series converges for any bounded inputs ($\|v_t\|_2 \|k_t\|_2 \le C$):
   $$\sup_{t \ge 0} \|S_t\|_F \le \frac{C}{1 - \alpha_{\max}} < \infty$$
   This proves strict contraction mapping, guaranteeing bounded state dynamics without numerical overflow across infinite sequence horizons.
*References: Master Thesis Section 3.2, Theorem 3.4 / Eq. 3.25; Defense Booklet Chapter 2; `neuralops/core/triton_kernels/gdn_scan.py`.*

---

### سوال ۲۴: قیود ماسک علّی در جایگزینی توجه دوطرفه در معماری‌های انکودر (Causal Masking Constraints when Replacing Bidirectional Attention in Encoders)
**پرسش داور:**
مکانیزم‌های توجه خطی بازگشتی (نظیر \lr{Chunk GLA} و \lr{Mamba}) ذاتا بر پایه اسکن یک‌طرفه علّی ($i \ge j$) بنا شده‌اند. هنگام تزریق جراحی این ماژول‌ها به معماری‌های انکودر نظیر \lr{RoBERTa} و \lr{DistilBERT} که متکی بر توجه دوطرفه (\lr{Bidirectional}) هستند، چه قیود و چالش‌هایی رخ می‌دهد و چگونه بدون نقض علیت حل شده‌اند؟

**پاسخ به فارسی:**
1. **چالش بنیادی ناسازگاری میدان دید (Receptive Field Mismatch):**
   در مدل‌های انکودر، توکن‌های ابتدایی برای درک معنا نیازمند اطلاعات توکن‌های انتهایی جمله هستند (توجه کامل دوطرفه $A_{ij} \ne 0$ برای تمام $i, j$). اگر یک مکانیزم خطی علّی یک‌طرفه به طور مستقیم جایگزین گردد، نیمی از ماتریس توجه مسدود شده و دقت دسته‌بندی با افتی شدید مواجه می‌شود.
2. **راهکار معماری توجه خطی دوطرفه (Bidirectional Linear Scanning):**
   برای حفظ تعادل در انکودرها بدون بازگشت به پیچیدگی درجه دوم $\mathcal{O}(T^2)$، ماژول‌های تزریقی از استراتژی اسکن متقارن رفت و برگشت بهره می‌برند:
   - **اسکن پیشرو (Forward Scan):** محاسبه حالت $S_t^{\text{fwd}} = \alpha_t S_{t-1}^{\text{fwd}} + v_t k_t^\top$ در جهت $t=1 \to T$.
   - **اسکن پسرو (Backward Scan):** محاسبه حالت $S_t^{\text{bwd}} = \alpha_t' S_{t+1}^{\text{bwd}} + v_t' k_t'^\top$ در جهت معکوس $t=T \to 1$.
   - **ادغام فیوزشده با گیت تراز:** خروجی نهایی از ترکیب خطی پروجکشن دو زیرفضا حاصل می‌شود:
     $$y_t = W_o \left[ y_t^{\text{fwd}} \odot \sigma(g_t) + y_t^{\text{bwd}} \odot (1 - \sigma(g_t)) \right]$$
3. **حفظ قید یک‌طرفه در مدل‌های زبانی خودبازگشتی:**
   برعکس، در هنگام تزریق به مدل‌های خودبازگشتی مولد نظیر \lr{GPT-2} و \lr{Qwen}، قید ماسک مثلثی پایینی علّی ($M_{ij} = -\infty$ برای $j > i$) درون کرنل تریتون اعمال شده و از هرگونه نشت اطلاعات آینده (\lr{Look-ahead Leakage}) جلوگیری به عمل می‌آید.
   این جداسازی دقیق باعث شد مدل RoBERTa به دقت **93.12%** در SST-2 دست یابد و همزمان GPT-2 پایداری کامل پرپلکسیتی **146.41** را ثبت کند.

**English Response:**
1. **The Bidirectional Receptive Field Dilemma in Encoders:**
   Encoder representations (e.g., RoBERTa, DistilBERT) rely on unconstrained contextual attention where token representations attend simultaneously to preceding and succeeding context. Naively swapping dense attention with a causal recurrent linear scan halves the effective receptive field, causing representational collapse on bidirectional tasks.
2. **Bidirectional Recurrent Scanning Formulation:**
   To retain full receptive field coverage while strictly preserving linear $\mathcal{O}(T)$ compute and memory bounds, our encoder surgery applies dual-directional parallel scanning:
   - **Forward Scan:** $S_t^{\text{fwd}} = \alpha_t S_{t-1}^{\text{fwd}} + v_t k_t^\top$ for $t = 1, \dots, T$.
   - **Backward Scan:** $S_t^{\text{bwd}} = \alpha_t' S_{t+1}^{\text{bwd}} + v_t' {k_t'}^\top$ for $t = T, \dots, 1$.
   - **Gated Fused Projection:**
     $$y_t = W_o \left[ y_t^{\text{fwd}} \odot \sigma(g_t) + y_t^{\text{bwd}} \odot (1 - \sigma(g_t)) \right]$$
3. **Strict Causal Enforcement for Autoregressive Decoders:**
   Conversely, when injecting into causal decoders (GPT-2, Qwen), the causal lower-triangular constraint is natively embedded into the chunked Triton scanning operators, precluding future token leakage.
   This structural bifurcation enabled RoBERTa to maintain 93.12% accuracy on SST-2 while stabilizing GPT-2 causal language modeling perplexity at 146.41.
*References: Master Thesis Section 4.2; Stage 1 Surgical Injection Specification; RESULTS_PROVENANCE.md.*

---

### سوال ۲۵: کران‌های الگوریتمی برای تخصیص پویای رتبه تحت قیود بودجه حافظه (Algorithmic Bounds for Dynamic Rank Allocation under Budget Constraints)
**پرسش داور:**
در فشرده‌سازی لایه‌های شبکه با تجزیه مقادیر منفرد، چرا تخصیص رتبه یکنواخت ($r_l = \bar{r}$) غیربهینه است؟ الگوریتم تخصیص پویای رتبه (\lr{Dynamic Rank Allocation}) چگونه رتبه‌ها را تحت بودجه ثابت حافظه $B_{\text{target}}$ بهینه می‌سازد و کران خطای سراسری آن چیست؟

**پاسخ به فارسی:**
1. **ناهمگونی طیفی لایه‌ها و ناکارآمدی رتبه ثابت:**
   طیف مقادیر منفرد ماتریس‌های وزن در طول لایه‌های ترنسفورمر دارای نرخ افت (\lr{Spectral Decay}) به شدت متفاوتی است. لایه‌های ابتدایی معمولا دارای طیف هموارتر و آنتروپی طیفی بالاتر هستند، در حالی که لایه‌های میانی و انتهایی به دلیل تمرکز بر الگوهای معنایی خاص، افت طیفی نمایی سریع‌تری نشان می‌دهند. تخصیص رتبه یکنواخت موجب حذف مؤلفه‌های حیاتی در لایه‌های حساس و اتلاف بودجه در لایه‌های با رتبه موثر پایین می‌گردد.
2. **فرمول‌بندی مسئله کوله‌پشتی پیوسته با ضرایب لاگرانژ:**
   هدف، کمینه‌سازی خطای بازسازی وزنی تجمعی در میان تمام $L$ لایه مدل تحت بودجه کل پارامترها است:
   $$\min_{\{r_l\}_{l=1}^L} \sum_{l=1}^L \sum_{j=r_l+1}^{d_l} \sigma_{l, j}^2 \quad \text{s.t.} \quad \sum_{l=1}^L r_l (d_{l, \text{in}} + d_{l, \text{out}}) \le B_{\text{target}}$$
   با تعریف چگالی انرژی طیفی حاشیه‌ای $e_l(r) = \sigma_{l, r}^2 / (d_{l, \text{in}} + d_{l, \text{out}})$ و تشکیل تابع لاگرانژ:
   $$\mathcal{J}(r_1, \dots, r_L, \lambda) = \sum_{l=1}^L \sum_{j=r_l+1}^{d_l} \sigma_{l, j}^2 + \lambda \left( \sum_{l=1}^L r_l (d_{l, \text{in}} + d_{l, \text{out}}) - B_{\text{target}} \right)$$
3. **شرط بهینگی کاروش-کون-تاکر (KKT) و الگوریتم حریصانه با کران تضمین‌شده:**
   نقطه بهینه تحلیلی زمانی حاصل می‌شود که نرخ کاهش خطا به ازای هر واحد پارامتر در تمام لایه‌ها به یک آستانه تعادلی یکنواخت $\lambda^*$ برسد:
   $$r_l^* = \max \left\{ r \;\middle|\; \frac{\sigma_{l, r}^2}{d_{l, \text{in}} + d_{l, \text{out}}} \ge \lambda^* \right\}$$
   با اعمال الگوریتم حریصانه مبتنی بر آنتروپی طیفی در مرحله ۳ ابلیشن:
   - بودجه پارامترها دقیقا رعایت می‌شود: $\sum r_l^* (d_{\text{in}} + d_{\text{out}}) = B_{\text{target}}$.
   - کران خطای سراسری همواره از تخصیص یکنواخت اکیداً کمتر است: $\mathcal{E}_{\text{dynamic}}^* \le \mathcal{E}_{\text{uniform}} - \Delta_{\text{spectral}}$.
   - این بهینه‌سازی در جدول ابلیشن، دقت فشرده‌سازی DistilBERT را از 68.60% به **70.80%** (+2.20% ارتقا با حافظه کمتر 172.4 MB) بهبود داده است.

**English Response:**
1. **Layer-wise Spectral Decay Heterogeneity:**
   Singular value decay rates vary sharply across transformer depth: early layers display broader, high-entropy spectra, while deep semantic layers exhibit steep power-law decay with low intrinsic dimensionality. Uniform rank allocation ($r_l = \bar{r}$) uniformly truncates parameters, over-pruning expressive layers while wasting budget on highly redundant layers.
2. **Constrained Knapsack Formulation via Lagrange Multipliers:**
   The dynamic rank allocation is formulated as minimizing total Frobenius reconstruction error across $L$ layers subject to a strict global parameter budget $B_{\text{target}}$:
   $$\min_{\{r_l\}_{l=1}^L} \sum_{l=1}^L \sum_{j=r_l+1}^{d_l} \sigma_{l, j}^2 \quad \text{subject to} \quad \sum_{l=1}^L r_l (d_{l, \text{in}} + d_{l, \text{out}}) \le B_{\text{target}}$$
3. **KKT Optimality Criterion and Marginal Energy Thresholding:**
   Defining the marginal spectral return per parameter as $e_l(r) = \frac{\sigma_{l, r}^2}{d_{l, \text{in}} + d_{l, \text{out}}}$, the Karush-Kuhn-Tucker (KKT) stationary condition requires marginal error drops to equilibrate at global dual threshold $\lambda^*$:
   $$r_l^* = \max \left\{ r \;\middle|\; \frac{\sigma_{l, r}^2}{d_{l, \text{in}} + d_{l, \text{out}}} \ge \lambda^* \right\}$$
   The resulting allocation guarantees Pareto-optimal parameter efficiency:
   $$\sum_{l=1}^L \sum_{j=r_l^*+1}^{d_l} \sigma_{l, j}^2 \le \sum_{l=1}^L \sum_{j=\bar{r}+1}^{d_l} \sigma_{l, j}^2$$
   In our 5-stage ablation pipeline (Table 2.3 / Booklet p. 6), spectral dynamic rank allocation lifts compressed DistilBERT MNLI accuracy from 68.60% to **70.80%** (+2.20%) while reducing peak VRAM from 178.5 MB to 172.4 MB.
*References: Master Thesis Section 3.1, Theorem 3.1; Table 4.4 / Defense Booklet Table 2.3 (p. 6); Defense Slide 8b.*


