---
name: neuralops-presentation-architect
description: "Author, verify, and orchestrate academic defense presentations, slide decks (Beamer, Marp, Google Slides), and oral delivery scripts for NeuralOps project defense and conference presentations"
version: "1.1.0"
author: "NeuralOps Team"
tags: ["neuralops", "slides", "presentation", "beamer", "defense", "marp", "google-slides", "project-day"]
---

# Skill: NeuralOps Defense Slide Architect & Oral Presentation Strategist

## When to use
Use this skill when designing, authoring, refining, verifying, or synchronizing official project defense slides, conference presentations, Project Day materials, and oral defense delivery scripts for NeuralOps. It ensures that every presentation slide:
1. Accurately showcases the full suite of **empirical achievements** of the project without omission.
2. Adheres strictly to University of Tehran Faculty of Engineering academic presentation guidelines.
3. Conforms to the 7-phase defense structure and timing synchronization in `academic/slides/DEFENSE_PRESENTATION_SCRIPT.md`.
4. Renders flawless bilingual typography (XePersian/Beamer or Marp) with mandatory English math-mode digits in all tables and equations.
5. Supports multi-format export across LaTeX Beamer (`academic/slides/slides.tex`, `defense_presentation.tex`), Marp Markdown (`academic/slides/slides.md`), and Google Slides (via `gslides`).

---

## Canonical Project Achievements to Showcase

Every defense presentation MUST highlight the verified quantitative and architectural achievements of NeuralOps:

### 1. Unified Architectural Scale & Scope
- **203 Unified Mathematical Modules**:
  - **131 Attention Mechanisms** registered across 7 architectural families (Exact Softmax, Recurrent SSM, Linear Attention, Low-Rank Approximations, Sparse Block Routing, Differential Attention, Hybrid/Interleaved).
  - **72 Model Compression Engines** registered across 5 paradigms (Low-Rank SVD, Pruning, Quantization, Distillation, and Hybrid).
  - Performance stratification: **42 PASS_FAST** modules achieving verified wall-clock speedup ($>1.0\times$) and **161 PASS_SLOW** modules providing certified mathematical baselines.

### 2. High-Performance Hardware Kernel Infrastructure
- **26 Canonical Triton Kernel Modules** in `neuralops/core/triton_kernels/` (housing 62 unified operators across 4 performance tiers).
- **114 Native CUDA C++ Kernel Implementations** (112 packages) in `neuralops/core/cuda_kernels/`.
- **3-Level Zero-Overhead Fallback Dispatch Hierarchy**:
  - *Level 1*: Native CUDA C++ / HIP compiled extension (maximum throughput).
  - *Level 2*: Triton JIT kernel (autotuned for FP16/BF16 with dynamic block tiling).
  - *Level 3*: Vectorized PyTorch reference (universal CPU/MPS fallback).
  - Dispatched via `@lru_cache` with zero allocations and mandatory warning emission on degraded fallback.

### 3. Empirical Hardware Bounds & Williams Roofline Modeling
- Benchmarked on **NVIDIA RTX 4070 (Ada Lovelace, SM 8.9)**:
  - Memory Bandwidth: $255.8\text{ GB/s}$ GDDR6.
  - Peak FP16 Tensor Core Compute: $70.6\text{ TFLOP/s}$.
  - Balance Knee Point: $I_{\text{knee}} = 276.0\text{ FLOP/Byte}$.
  - **Fundamental Empirical Finding**: Standard Softmax Attention operates at $I \le 183.1\text{ FLOP/Byte}$, demonstrating that long-context attention is strictly **DRAM Bandwidth Bound (Memory-Bound)**, which justifies the necessity of SVD rank compression and linear recurrence.

### 4. Surgical Factorization & Knowledge Distillation Healing
- **4-Stage Model Surgery Pipeline**: Decomposition $\to$ In-Place Module Transplant $\to$ Riemannian Manifold Retraction $\to$ Ridge Calibration Healing.
- **6-Pillar Distillation Recovery Bundle**: Restores accuracy from catastrophic post-truncation collapse ($51.30\%$) back to **$89.40\%$** on SST-2 and $73.62\%$ on RoBERTa-base, achieving near-lossless retention at $2.0\times$ parameter compression.
- **HNC-SVD Adaptive Rank Selection**: Spectral entropy optimization identifying the $r=32$ knee-point as the optimal Pareto frontier across accuracy, latency, and VRAM.
- **Long-Context Linear Scaling**: Empirical crossover at $N \approx 2048$ tokens, scaling linearly up to $131\text{k}$ tokens with constant KV-cache memory footprint.

---

## Detailed Slide-by-Slide Defense Presentation Blueprint (24 Dedicated Slides)

Every single slide in the master defense deck MUST be dedicated to a concrete technical advance, mathematically formulated, backed by empirical data, and paired with a publication-grade figure:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           7-PHASE DEFENSE TIMELINE                               │
├────────────────────┬──────────────┬──────────┬───────────────────────────────────┤
│ Phase              │ Slide Range  │ Duration │ Key Objectives                    │
├────────────────────┼──────────────┼──────────┼───────────────────────────────────┤
│ 1. Opening & Map   │ Slides 1–2   │ 2 mins   │ Welcome, Committee, Agenda        │
│ 2. Foundations     │ Slides 3–5   │ 5 mins   │ Memory Wall O(N^2), Cholesky, SVD │
│ 3. SOTA Gaps       │ Slides 6–8   │ 8 mins   │ 5 SOTA Gaps Closed, 4-Stage Surg  │
│ 4. 6-Pillar KD     │ Slides 9–11  │ 7 mins   │ 6-Pillar Distillation, 203 Taxon  │
│ 5. Pareto Scaling  │ Slides 12–15 │ 6 mins   │ Top-15 Accuracy, 3D Pareto, 131k  │
│ 6. Hardware & GPU  │ Slides 16–21 │ 5 mins   │ 26 Triton, 114 CUDA, Roofline     │
│ 7. Synthesis & Q&A │ Slides 22–24 │ 2 mins   │ Key Takeaways, Committee Q&A      │
└────────────────────┴──────────────┴──────────┴───────────────────────────────────┘
```

---

### Slide 1: Title & Examination Committee
- **Slide Title**: عنوان رساله و کمیتهٔ داوری
- **Concrete Scope**: معرفی رسمی جلسه دفاع، دانشجو، استاد راهنما، و دانشکدهٔ مهندسی برق و کامپیوتر دانشگاه تهران.
- **Visual Artifacts**: آرم رسمی دانشگاه تهران (`eng-logo.png`, `logo.png`).
- **Exact Meta**: نگارنده: محمد طاها مجلسی (۸۱۰۱۰۱۵۰۴) | استاد راهنما: دکتر احمد کلهر | شهریور/آبان ۱۴۰۵.
- **Speaker Cue**: خوش‌آمدگویی رسمی، بیان هدف پژوهش در پیوند زدن مبانی ریاضی آنالیز طیفی به معماری سطح پایین کارت‌های گرافیک.

---

### Slide 2: Strategic Research Roadmap
- **Slide Title**: نقشهٔ راهبردی و سرفصل‌های ارائه دفاع
- **Concrete Scope**: تقسیم‌بندی ۳۵ دقیقه‌ای ارائه در ۷ فاز عملیاتی از مبانی تئوری تا شتاب‌دهی سخت‌افزاری.
- **Visual Artifacts**: دیاگرام جریان کار ۷ فازی با بسته‌های زمانی دقیق (۲ تا ۳۵ دقیقه).
- **Key Milestones**: دیوار حافظه $\to$ لم چولسکی $\to$ خط‌لوله ۴ مرحله‌ای $\to$ تقطیر ۶ ستونی $\to$ تله‌متری ۲۰۳ ماژول $\to$ کرنل‌های Triton/CUDA $\to$ دفاعیه.

---

### Slide 3: The Memory Wall & $\mathcal{O}(N^2)$ Complexity Bottleneck
- **Slide Title**: بحران دیوار حافظه (Memory Wall) و پیچیدگی درجه‌دوم $\mathcal{O}(N^2)$
- **Concrete Scope**: تحلیل گلوگاه محاسباتی و حافظه‌ای خودتوجهی استاندارد (Softmax Self-Attention).
- **Mathematical Formulation**:
  $$\mathbf{A} = \operatorname{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}, \quad \text{Memory Traffic} = \mathcal{O}(N^2 d) + \mathcal{O}(L \cdot B \cdot N \cdot d_{\text{kv}})$$
- **Empirical Evidence**: در طول توالی $N > 32\text{k}$، مادی‌سازی ماتریس توجه از مرجع VRAM کل کارت ($12\text{ GB}$) تجاوز می‌کند؛ استنتاج خودبازگشتی کاملاً DRAM Bandwidth-Bound می‌شود.
- **Visual Artifacts**: نمودار انفجار کش کلید-مقدار (`kv_cache_latent_memory_scaling.png`).
- **Speaker Cue**: تشریح تفاوت حالت Training با رژیم Autoregressive Decoding و ضرورت توجه خطی و فشرده‌سازی SVD.

---

### Slide 4: Mathematical Foundations: Cholesky Covariance Whitening
- **Slide Title**: سفیدسازی کوواریانس با تجزیه چولسکی و قضیه اکرت-یانگ-میرسکی
- **Concrete Scope**: اثبات ریاضی بهینگی فاکتورسازی SVD در حضور ماتریس کوواریانس فعال‌سازی‌های ورودی.
- **Mathematical Formulation**:
  $$\min_{\mathbf{W}_r} \|(\mathbf{W} - \mathbf{W}_r)\mathbf{H}_X^{1/2}\|_F^2, \quad \mathbf{H}_X = \mathbf{L}\mathbf{L}^\top, \quad \widetilde{\mathbf{W}} = \mathbf{W}\mathbf{L} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$
  $$\mathbf{W}_r = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^\top \mathbf{L}^{-1}$$
- **Theoretical Guarantee**: قضیه ۱ اثبات می‌کند که برش در فضای سفیدشده، خطای اکرت-یانگ را کمینه کرده و مانع از کج‌شدگی بعدهای با واریانس بالا می‌شود.
- **Speaker Cue**: بیان این نکته که SVD استاندارد توزیع داده‌ها را نادیده می‌گیرد، در حالی که سفیدسازی چولسکی فضای بازنمایی را کاملاً ایزوتروپیک می‌سازد.

---

### Slide 5: Empirical Optimality: SVD-LLM v2 vs Naive Truncation
- **Slide Title**: اعتبارسنجی تجربی بهینگی: مقایسه SVD-LLM v2 با برش خام SVD
- **Concrete Scope**: نشان دادن افت فاجعه‌بار برش سنتی در برابر خطای نزدیک به صفر روش بهینه‌سازی‌شده.
- **Empirical Data**: در نرخ فشرده‌سازی $2.0\times$، خطای فروبنیوس در SVD سنتی به بیش از $0.45$ می‌رسد، در حالی که در SVD-LLM v2 خطا زیر $0.042$ کنترل می‌شود.
- **Visual Artifacts**: نمودار برداری مقایسه نرخ خطای فروبنیوس (`svd_llm_v2_truncation_optimality.png`).
- **Speaker Cue**: هدایت توجه داوران به همگرایی خطای سفیدسازی چولسکی به صفر حتی در رتبه‌های کمینه ($r=32$).

---

### Slide 6: Closing 5 SOTA Theoretical Gaps
- **Slide Title**: بسته‌شدن ۵ شکاف علمی مرز دانش (Closing 5 SOTA Gaps)
- **Concrete Scope**: تشریح پنج نوآوری نظری و معماری منحصربه‌فرد NeuralOps:
  1. *WGQA Simplex Stability*: اثبات پایداری گرادیان در گیت‌بندی سیمپلکس (قضیه ۷).
  2. *ASVD KV-Streaming*: استریمینگ بلادرنگ بافر کش بدون وقفه در تخصیص حافظه.
  3. *FP8 Dynamic Range Bound*: محدودسازی خطای کوانتش FP8 E4M3 در تانسورهای خروجی (قضیه ۳).
  4. *Dynamic Sparse Attention (DSA)*: آدرس‌دهی متغیر بلوک‌های تنک با سقف حافظه ثابت.
  5. *Residual-Enhanced Adaptive Pruning (REAP)*: حفظ پایه‌های فرعی از طریق جبران باقی‌مانده طیفی.
- **Speaker Cue**: اثبات این که NeuralOps تنها یک بنچ‌مارک نیست، بلکه فرمولاسیون‌های پایه‌ای جدید به ادبیات توجه و فشرده‌سازی افزوده است.

---

### Slide 7: The 4-Stage Model Surgery Pipeline
- **Slide Title**: خط لوله ۴ مرحله‌ای جراحی مدل بدون بازآموزی کلان
- **Concrete Scope**: متدولوژی جایگزینی درون‌جای ماژول‌های ترنسفورمر روی کارت‌های گرافیکی مصرف‌کننده.
- **The 4 Stages**:
  1. *Decomposition*: اسکن گراف مدل، استخراج ماتریس‌های پروجکشن $W_q, W_k, W_v, W_o$.
  2. *In-Place Replacement*: تزریق ماژول هدف از رجیستری با حفظ ساختار تانسوری و قرارداد خروجی.
  3. *Manifold Retraction*: استفاده از رترکشن مانیفولد ریمانی استیفل برای حفظ تعامد پایه‌های تبدیل.
  4. *Ridge Healing*: جبران جابه‌جایی توزیع (Distribution Shift) با رگرسیون ریج در لایه‌های منجمد میانی.
- **Visual Artifacts**: دیاگرام گام‌به‌گام جراحی (`fig_ablation_pipeline_5step.pdf`).

---

### Slide 8: Step-by-Step Ablation Restoration ($51.30\% \to 89.40\%$)
- **Slide Title**: بازیافت گام‌به‌گام دقت در ابلیشن ۵ مرحله‌ای جراحی
- **Concrete Scope**: ردیابی آزمایشگاهی بازیابی دقت در مدل‌های DistilBERT و RoBERTa-base بر روی بنچ‌مارک SST-2.
- **Empirical Progression Table**:
  $$\text{Dense Baseline: } 90.25\% \longrightarrow \text{Raw Truncation: } 51.30\% \longrightarrow \text{Cholesky Whitening: } 78.40\%$$
  $$\longrightarrow \text{Ridge Healing: } 84.60\% \longrightarrow \text{LoRA Adaptation: } 86.80\% \longrightarrow \text{Final 6-Pillar KD: } \mathbf{89.40\%}$$
- **Visual Artifacts**: نمودار ستونی ۵ مرحله‌ای ابلیشن جراحی (`fig_ablation_pipeline_5step.pdf`).
- **Speaker Cue**: تشریح بازیابی خیره‌کننده بیش از ۳۸ درصد دقت بدون حتی یک اپوک آموزش کامل مدل مبنا.

---

### Slide 9: The 6-Pillar Knowledge Distillation (KD) Architecture
- **Slide Title**: بسته ۶ ستونی تقطیر دانش (6-Pillar Knowledge Distillation)
- **Concrete Scope**: معماری جامع تابع زیان تلفیقی برای کالیبراسیون و آموزش جبرانی:
  $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{task}} + \beta \mathcal{L}_{\text{logits}} + \gamma \mathcal{L}_{\text{repr}} + \delta \mathcal{L}_{\text{attn}} + \epsilon \mathcal{L}_{\text{cosine}} + \zeta \mathcal{L}_{\text{spectral}}$$
- **Mechanisms**:
  - *Layerwise Cosine Pullback*: هم‌راستاسازی زاویه‌ای بردار فعال‌سازی‌ها.
  - *Attention Map KL*: تطابق ماتریس وزن‌های احتمالاتی توجه.
  - *Residual Spectral Matching*: حفظ آنتروپی مقادیر منفرد باقی‌مانده.
- **Visual Artifacts**: نمودار آبشاری بازیابی ستون‌های تقطیر (`fig_6pillar_ablation.pdf`).

---

### Slide 10: Theorem 10: Distillation Contraction & Layer Transfer Fidelity
- **Slide Title**: قضیه ۱۰: انقباض هندسی خطای تقطیر و ماتریس همبستگی لایه‌ای
- **Concrete Scope**: اثبات ریاضی همگرایی خطای جراحی و نمایش تجربی وفاداری انتقال دانش میان لایه‌ها.
- **Mathematical Statement**: اثبات انقباض متوالی خطا $\|\mathbf{E}_{l+1}\|_F \le \rho \|\mathbf{E}_l\|_F$ با ضریب انقباض $\rho < 1$ تحت لایه‌های لیپ‌شیتز نرمالایزیشن.
- **Visual Artifacts**: ماتریس هیت‌مپ همبستگی کسینوسی $6 \times 6$ لایه‌ها (`fig_6pillar_heatmap.pdf`).
- **Speaker Cue**: اشاره به پایداری انتقال گرادیان در تمام لایه‌های ترنسفورمر بدون واگرایی یا اشباع فعال‌سازی‌ها.

---

### Slide 11: Master Taxonomy: 203 Modules Across 12 Paradigms
- **Slide Title**: تاکسونومی جامع ۲۰۳ ماژول: ۱۳۱ مکانیزم توجه و ۷۲ موتور فشرده‌سازی
- **Concrete Scope**: دسته‌بندی علمی کلان اکوسیستم ثبت‌شده در پلتفرم NeuralOps.
- **Attention Taxonomy (131 Modules)**:
  - Exact Softmax (FlashAttention-2/3, SDPA): 18 ماژول
  - Recurrent SSM (Mamba, Gated DeltaNet, RWKV): 24 ماژول
  - Linear Attention (Cosformer, Linformer, Performer): 28 ماژول
  - Low-Rank Approximations (DeepSeek MLA, LoRA-Attn): 22 ماژول
  - Sparse Block Routing (NSA, BigBird, Longformer): 19 ماژول
  - Differential & Hybrid: 20 ماژول
- **Compression Taxonomy (72 Modules)**: Low-Rank (28), Pruning (16), Quantization (14), Distillation (14).
- **Visual Artifacts**: نمودار راداری ۵ بعدی مقایسه خانواده‌ها (`chart_radar_families.png`).

---

### Slide 12: Top-15 Empirical Accuracy Leaderboard
- **Slide Title**: جدول برترین‌های دقت (Top-15 Accuracy Leaderboard)
- **Concrete Scope**: رتبه‌بندی عملکرد ماژول‌های توجه و فشرده‌سازی روی تسک‌های استاندارد GLUE (SST-2, MRPC, CoLA).
- **Empirical Table (English Math-Mode)**:
  | رتبه | مکانیزم / ماژول | خانواده | دقت SST-2 | نرخ حفظ دقت | شتاب عملیاتی |
  |:---:|:---|:---|:---:|:---:|:---:|
  | $1$ | \lr{\texttt{flash\_attention\_3}} | Exact Fused | $90.25\%$ | $100.0\%$ | $3.24\times$ |
  | $2$ | \lr{\texttt{gated\_delta\_net}} | Recurrent SSM | $89.80\%$ | $99.50\%$ | $2.45\times$ |
  | $3$ | \lr{\texttt{hnc\_svd\_r32}} | Low-Rank SVD | $89.40\%$ | $99.06\%$ | $1.74\times$ |
  | $4$ | \lr{\texttt{cosformer\_triton}} | Linear | $88.90\%$ | $98.50\%$ | $2.12\times$ |
  | $5$ | \lr{\texttt{deepseek\_mla}} | Low-Rank | $88.60\%$ | $98.17\%$ | $1.68\times$ |
- **Visual Artifacts**: نمودار رتبه‌بندی ۱۵ ماژول برتر (`chart_top15_accuracy.png`).

---

### Slide 13: 3D Multi-Objective Pareto Frontier
- **Slide Title**: جبهه بهینگی سه‌بعدی پارتو (دقت، تأخیر و حافظه VRAM)
- **Concrete Scope**: ارزیابی تعادل چندهدفه برای استقرار صنعتی مدل‌ها در شرایط محدودیت منابع سخت‌افزاری.
- **Pareto Invariants**: هیچ ماژولی در خارج از این جبهه بدون افت یکی از ابعاد (دقت، سرعت، یا VRAM) قابل انتخاب نیست.
- **Visual Artifacts**: نمودار جبهه پارتو در برابر VRAM و Throughput (`chart_pareto_vram.png`, `pareto_accuracy_vs_throughput.png`).
- **Speaker Cue**: معرفی ماژول‌های بهینه به عنوان نامزدهای طلایی جایگزینی برای تجهیزات لبه (Edge Devices) و سرورهای ابری.

---

### Slide 14: SVD Rank Sensitivity: The $r=32$ Knee-Point Curve
- **Slide Title**: تحلیل حساسیت رتبه SVD و نقطهٔ زانویی بهینه ($r=32$)
- **Concrete Scope**: کالبدشکافی آزمایشگاهی رتبه‌های فاکتورسازی $r \in [4, 128]$ مطابق داده‌های جدول ۴-۱۷ رساله.
- **Empirical Metrics**:
  - $r=4$: دقت $71.20\%$ (افت شدید بازنمایی، فروپاشی طیفی).
  - $r=16$: دقت $84.50\%$ (بهبود اولیه اما ناپایدار در جملات پیچیده).
  - $r=32$: دقت $\mathbf{89.40\%}$ با شتاب $1.74\times$ و کاهش $50\%$ وزن‌ها (**نقطه زانویی بهینه**).
  - $r=64$: دقت $89.90\%$ (افزایش ناچیز دقت در ازای افت شدید شتاب به $1.22\times$).
- **Visual Artifacts**: منحنی زانویی رتبه SVD (`fig_rank_sweep_adasvd.pdf`).
- **Speaker Cue**: اثبات این که چرا رتبه ۳۲ تعادل بی‌نقص ریاضی و شتاب سخت‌افزاری را برقرار می‌سازد.

---

### Slide 15: Long-Context Scaling: The $N=2048$ Crossover & 131k Frontier
- **Slide Title**: مقیاس‌پذیری طول توالی: نقطه تقاطع $N=2048$ و افق ۱۳۱ هزار توکن
- **Concrete Scope**: ارزیابی مقایسه‌ای زمان اجرا و مصرف حافظه از $N=512$ تا $N=131072$ توکن.
- **Scaling Dynamics**:
  - در $N < 2048$: توجه استاندارد به دلیل ثابت‌های مقیاس ضرب ماتریسی سریع‌تر است.
  - در $N = 2048$: نقطه تقاطع تجربی (Empirical Crossover)؛ خطای تأخیر یکسان می‌شود.
  - در $N > 2048$: توجه خطی $\mathcal{O}(N)$ رفتار مقتدرانه خود را تحمیل می‌کند؛ در $N=32\text{k}$ شتاب بیش از $4.5\times$ و در $N=131\text{k}$ بیش از $12.8\times$ ثبت می‌شود.
- **Visual Artifacts**: نمودار لگاریتمی مقیاس‌پذیری طول توالی (`sequence_length_scaling.png`).

---

### Slide 16: High-Throughput Kernel Stack: 26 Triton & 114 CUDA Implementations
- **Slide Title**: پشته شتاب‌دهی سخت‌افزاری: ۲۶ ماژول Triton و ۱۱۴ بسته بومی CUDA
- **Concrete Scope**: مهندسی کرنل‌های سطح پایین در پلتفرم NeuralOps جهت استفاده از تمام پهنای باند تراشه.
- **Kernel Stratification**:
  - ۲۶ ماژول Triton JIT در ۴ رده عملیاتی (Triton Core, Fused Attn, Linear Recurrence, Manifold Surgery).
  - ۱۱۴ بسته کد نیتیو C++/CUDA برای اپراتورهای فشرده‌سازی و تجزیه طیفی ماتریس‌ها.
- **Visual Artifacts**: لیدربورد شتاب کرنل‌های تریتون نسبت به پایتورچ (`triton_speedup_leaderboard.png`).
- **Speaker Cue**: تأکید بر فیوز کردن عملیات در رجیسترهای SRAM جهت حذف تبادل داده‌های اضافه با حافظه سراسری DRAM.

---

### Slide 17: Williams Hardware Roofline Modeling on SM 8.9 (RTX 4070)
- **Slide Title**: تحلیل سقف محاسباتی ویلیامز (Williams Roofline) روی معماری Ada Lovelace
- **Concrete Scope**: سنجش شدت محاسباتی (Arithmetic Intensity) ماژول‌ها در برابر مرزهای فیزیکی کارت گرافیک.
- **Hardware Parameters**:
  - پهنای باند حافظه: $B_{\text{peak}} = 255.8\text{ GB/s}$
  - سقف محاسباتی ممیز شناور FP16: $P_{\text{peak}} = 70.6\text{ TFLOP/s}$
  - نقطه زانویی توازن سخت‌افزار: $I_{\text{knee}} = 276.0\text{ FLOP/Byte}$
  - **مرز تجربی توجه استاندارد**: $I_{\text{SDPA}} = 183.1\text{ FLOP/Byte} \ll I_{\text{knee}}$
- **Physical Conclusion**: تمامی مکانیزم‌های توجه استاندارد در رژیم Memory-Bound عمل می‌کنند و تنها با تقلیل حافظه به ازای هر توکن می‌توان شتاب واقعی به دست آورد.
- **Visual Artifacts**: نمودار رسمی روف‌لاین ویلیامز (`fig_roofline_hardware.pdf`).

---

### Slide 18: Microarchitectural Forensics & Hardware Safety Laws
- **Slide Title**: کالبدشکافی عیوب ریزمعماری و ۶ قانون ایمنی گسیل (Safety Laws)
- **Concrete Scope**: تحلیل دو باگ سخت‌افزاری واقعی در کرنل‌ها و استخراج قوانین محافظتی:
  1. *L2 Cache Overflow in Chunk GLA*: ردپای حافظه $48\text{ MB}$ از حافظه نهان L2 کارت ($36\text{ MB}$) فراتر رفته و موجب ریزش داده به DRAM و افت شدید سرعت شد. راهکار: تایل‌بندی پویا بر اساس ظرفیت L2.
  2. *Register Spilling in FP8 GEMM*: مصرف ۸۰ ثبات به ازای هر ترد موجب کاهش تصدی وارپ (Warp Occupancy) شد. راهکار: بازآرایی پرولوگ حلقه و کش کردن در شرد مموری.
- **6 Dispatch Safety Laws**: تضمین صفر بودن خطاهای خاموش و بازگشت خودکار به مسیرهای امن پایتورچ در صورت بروز ناپایداری عددی.

---

### Slide 19: End-to-End Latency Comparison: 3-Level Dispatch Hierarchy
- **Slide Title**: مقایسه سه‌سطحی زمان اجرا: نیتیو CUDA، تریتون و پایتورچ
- **Concrete Scope**: بنچ‌مارک هم‌زمان ۱۸ اپراتور منتخب در هر سه سطح پشته نرم‌افزاری.
- **Empirical Speedup Metrics**:
  - کرنل‌های Level 1 (CUDA): شتاب میانگین $2.84\times$ تا $4.15\times$.
  - کرنل‌های Level 2 (Triton): شتاب میانگین $1.95\times$ تا $3.20\times$.
  - مسیر Level 3 (PyTorch Reference): خطای نسبی صفر (Ground Truth Baseline).
- **Visual Artifacts**: نمودار میله‌ای مقایسه تاخیر در سه سطح (`three_level_latency_comparison.png`).

---

### Slide 20: Pre-Training vs Surgery: Resource & Cost Efficiency
- **Slide Title**: کارایی منابع و انرژی: مقایسه جراحی مدل در برابر آموزش مجدد
- **Concrete Scope**: ارزیابی اقتصادی و محاسباتی متدولوژی NeuralOps برای جامعه پژوهشی.
- **Comparative Metrics**:
  - *Full Retraining*: نیازمند ۱۲۸ تا ۵۱۲ کارت A100، صدها ساعت زمان و هزاران دلار هزینه برق.
  - *NeuralOps Model Surgery*: انجام کالیبراسیون و بازیابی کامل در کمتر از **۲ ساعت** روی **یک کارت تکی RTX 4070** با توان مصرفی کمتر از ۲۰۰ وات!
- **Speaker Cue**: تأکید بر دموکراتیزه کردن پژوهش‌های هوش مصنوعی برای دانشگاه‌ها و پژوهشگران مستقل بدون دسترسی به کلاسترهای ابری عظیم.

---

### Slide 21: Live Software Architecture & Python CLI Verification
- **Slide Title**: معماری نرم‌افزاری، اکوسیستم متن‌باز و رابط‌های خط فرمان (CLI)
- **Concrete Scope**: تشریح استانداردهای نرم‌افزاری پلتفرم، بارگذاری تنبل (PEP 562 Lazy Loading)، تست‌های خودکار و CLI رسمی:
  ```bash
  neuralops benchmark --attention cosformer --seq-len 4096 --backend triton
  neuralops surgery --model roberta-base --compressor asvd --rank 32 --heal-steps 200
  ```
- **Code Quality Metrics**: بیش از ۲۵۰ آزمون واحد و پوشش یکپارچگی، کاملاً هماهنگ با PyTorch 2.x و کامپایلر Triton.

---

### Slide 22: Primary Research Contributions
- **Slide Title**: جمع‌بندی مشارکت‌ها و دستاوردهای بنیادین رساله
- **Concrete Scope**: فهرست صلب ۶ دستاورد علمی و مهندسی پروژه:
  1. پلتفرم یکپارچه با ۲۰۳ ماژول ریاضی (۱۳۱ توجه، ۷۲ فشرده‌ساز) با دروازه گسیل ۳ سطحی.
  2. اثبات ریاضی و اعتبارسنجی تجربی سفیدسازی چولسکی و قضیه بهینگی اکرت-یانگ-میرسکی.
  3. توسعه خط‌لوله ۴ مرحله‌ای جراحی بدون بازآموزی کلان با بازیابی دقت تا $89.40\%$.
  4. بسته ۶ ستونی تقطیر دانش و اثبات قضیه انقباض خطای گرادیان (Theorem 10).
  5. پیاده‌سازی و انتشار ۲۶ ماژول Triton JIT و ۱۱۴ بسته نیتیو CUDA C++ با شتاب تا $4.15\times$.
  6. ارزیابی جامع سقف محاسباتی روف‌لاین و کشف عیوب ریزمعماری در پردازش طول دنباله تا ۱۳۱ هزار توکن.

---

### Slide 23: Open-Source Artifacts & Artifact Availability
- **Slide Title**: دسترسی‌پذیری، تکرارپذیری آزمایش‌ها و انتشارات علمی
- **Concrete Scope**: تبیین تکرارپذیری ۱۰۰٪ یافته‌های رساله:
  - مخزن رسمی متن‌باز گیتهاب: `https://github.com/tahamajs/NeuralOps`
  - لجر مستر نتایج تجربی: `results/registry/master_ledger.jsonl` (بیش از ۵۰۰ ران رسمی).
  - پایان‌نامه جامع کارشناسی (۳۳۲ صفحه) تدوین‌شده با زی‌پرشین بر اساس شیوه‌نامه دانشگاه تهران.
  - اسلایدهای ارائه رسمی و کدهای بازتولید نمودارها.

---

### Slide 24: Conclusion & Committee Q&A Session
- **Slide Title**: نتیجه‌گیری نهایی و گشایش نشست پرسش و پاسخ
- **Concrete Scope**: قدردانی از استاد راهنما و هیئت محترم داوران دانشگاه تهران و اعلام آمادگی برای دفاع علمی.
- **Visual Elements**: اطلاعات تماس داوطلب، بارکد دسترسی به مخزن پروژه، آرم دانشگاه تهران.
- **Speaker Cue**: «با سپاس فراوان از توجه و شکیبایی اساتید بزرگوار و داوران ارجمند، با کمال افتخار آماده پاسخگویی به پرسش‌ها و نقدهای علمی شما هستم.»

---

## Strict Slide Formatting & Typography Standards

### 1. English Math-Mode Digits in All Tables and Metrics
- **Strict Law**: All digits, percentages, speedup ratios, latencies, ranks, and dimensions displayed in slide tables or metric boxes MUST be written in English digits in math mode `$ ... $`:
  ```latex
  % Banned (Persian digits cause punctuation flipping in Beamer/XePersian):
  ۳۴.۸۰٪ & ۷۴.۱۰٪ & ۱.۷۴x

  % Mandatory Standard:
  $34.80\%$ & $74.10\%$ & $1.74\times$
  ```
- English model names and identifiers must be wrapped in `\lr{\texttt{...}}` (e.g. `\lr{\texttt{RoBERTa-base}}`, `\lr{\texttt{DistilBERT}}`).

### 2. High-Impact Visual Analytics Standard
- Every content slide MUST contain at least one clear visual artifact:
  - Vector PDF charts from `academic/thesis/figures/` (or raster 300 DPI from `academic/slides/`).
  - System diagrams from `academic/thesis/img/`.
- Never produce text-only slides. A slide with pure text and no diagram or metric card is considered unfinished.

### 3. Slide Layout Density Constraints
- Maximum 4–5 bullet points per slide.
- Use bold lead-in phrases for every bullet point.
- Use structured Beamer color boxes (`\begin{block}`, `\begin{alertblock}`, `\begin{exampleblock}`) with the official University of Tehran palette:
  - Primary Blue: `#0D3270`
  - Gold Accent: `#BE9123`
  - Soft Blue / Gold / Green background fills for readability.

### 4. Zero Overfull \hbox in Beamer Tables
- Tables inside slides must use `\small` or `\footnotesize` and `tabularx` with `X` columns to prevent clipping beyond the 16:9 frame boundary.
- Always set `\setlength{\tabcolsep}{3.5pt}` and `\renewcommand{\arraystretch}{1.15}` for crisp, uncompressed table formatting.

---

## Defense Q&A Preparation Matrix (Top 5 Anticipated Inquiries)

| Committee Question | Core Scientific Defense | Primary Empirical Evidence |
|:---|:---|:---|
| **Q1**: Why is SVD-LLM v2 superior to standard truncated SVD? | Standard SVD truncates based on raw weight magnitudes without accounting for activation variance. Cholesky whitening weights the singular vectors by empirical input covariance $\mathbf{H}_X$, strictly minimizing output perturbation per the Eckart-Young-Mirsky theorem. | Slide 5; `svd_llm_v2_truncation_optimality.png`; Theorem 1. |
| **Q2**: Why does standard attention fail at long sequences on Ada Lovelace? | Roofline analysis shows attention has arithmetic intensity $I \le 183.1\text{ FLOP/Byte}$, well below SM 8.9 balance knee ($276.0\text{ FLOP/Byte}$). It is strictly DRAM bandwidth bound. Loading KV-cache stalls SM warps. | Slide 17; `fig_roofline_hardware.pdf`; Table 3.2. |
| **Q3**: How does the 6-Pillar KD prevent catastrophic forgetting? | Truncation drops accuracy to $51.30\%$. 6-Pillar KD introduces layerwise cosine pullback and residual spectral matching, contracting the gradient error and restoring accuracy to $89.40\%$. | Slide 9; `fig_ablation_pipeline_5step.pdf`; Theorem 10. |
| **Q4**: Why did you implement both CUDA C++ and Triton JIT kernels? | Triton JIT provides autotuning and rapid prototyping for 26 unified modules, while 114 native CUDA C++ kernels eliminate Python interpreter overhead and allow warp-level shuffles and TMA transfers. | Slide 16; `triton_speedup_leaderboard.png`; Section 3.2. |
| **Q5**: What justifies the $r=32$ rank knee-point? | At $r < 32$, singular value energy retention drops sharply below $85\%$, causing unrecoverable downstream accuracy loss. At $r=32$, accuracy recovery reaches $89.40\%$ with $2.0\times$ parameter savings. | Slide 14; `fig_rank_sweep_adasvd.pdf`; Table 4.17. |

---

## Pre-Defense Presentation Checklist

Before delivering the presentation, verify:
- [ ] 16:9 Aspect Ratio enabled (`aspectratio=169`).
- [ ] All 203 modules, 131 attentions, 72 compressors, 26 Triton kernels, and 114 CUDA packages accurately stated.
- [ ] Every slide has at least one high-resolution vector chart or architectural diagram.
- [ ] All table numbers, speedups, and dimensions are in English math mode `$ ... $`.
- [ ] Zero `Overfull \hbox` or clipped text on any slide frame.
- [ ] Oral delivery script in `academic/slides/DEFENSE_PRESENTATION_SCRIPT.md` perfectly matches slide sequence.
- [ ] Timing strictly aligns with 35-minute allocation (plus 15-minute Q&A).
