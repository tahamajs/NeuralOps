import os, re, glob

BASE_DIR = '/Users/tahamajs/Documents/uni/Project'
MODULES_DIR = os.path.join(BASE_DIR, 'thesis_final/attentions_latex/individual_modules')
CHAPTER_FILE = os.path.join(BASE_DIR, 'thesis_final/chapter_attentions.tex')

# 1. Load curated 100 list
local_vars = {}
with open(os.path.join(BASE_DIR, 'thesis_final/build_curated_100_list.py')) as f:
    exec(f.read(), {}, local_vars)
curated_100 = local_vars['curated_100']
assert len(curated_100) == 100, f"Expected 100 modules, got {len(curated_100)}"

# 2. Architectural mapping for user-provided diagrams
ARCH_MAP = {
    'full_attention': {
        'img': 'img/arch_full_attention.jpeg',
        'caption': r'نمودار گردش داده و معماری تانسوری مکانیزم توجه چندسره استاندارد \lr{(Full Multi-Head Attention)} به همراه انطباق \lr{LoRA} و پروجکشن‌های خطی'
    },
    'fusion_attn': {
        'img': 'img/arch_fusion_attn.jpeg',
        'caption': r'معماری هیبریدی ماژول \lr{FusionAttention} با تفکیک مسیر سریع برای توالی‌های کوتاه \lr{($N \le 128$)} و توجه تنک بلوکی \lr{(Top-K Block Selection)} برای توالی‌های بلند'
    },
    'primal_attention': {
        'img': 'img/arch_primal_attention.jpeg',
        'caption': r'نمودار ساختار تانسوری ماژول توجه بدوی \lr{(Primal Attention)} و فاکتورسازی فضای ویژگی‌ها'
    },
    'gqa_attention': {
        'img': 'img/arch_grouped_query_attention.jpeg',
        'caption': r'معماری توجه با پرس‌وجوی گروه‌بندی‌شده \lr{(Grouped-Query Attention - GQA)} جهت بهینه‌سازی حافظه \lr{KV-Cache}'
    },
    'wgqa_attention': {
        'img': 'img/arch_grouped_query_attention.jpeg',
        'caption': r'معماری توجه با پرس‌وجوی گروه‌بندی‌شده وزنی \lr{(Weighted GQA)} با بازآرایی سرهای کلید و مقدار'
    },
    'adaptive_sparse_attention': {
        'img': 'img/arch_adaptive_sparse_attention.jpeg',
        'caption': r'معماری تانسوری توجه تنک تطبیقی \lr{(Adaptive Sparse Attention)} با گزینش پویای الگوهای توجه'
    },
    'adaptive_span_attention': {
        'img': 'img/arch_adaptive_span_attention.jpeg',
        'caption': r'سازوکار تنظیم پویای طول پنجره زمینه در ماژول \lr{Adaptive Span Attention}'
    },
    'adasplash_attention': {
        'img': 'img/arch_adasplash_attention.jpeg',
        'caption': r'معماری محاسباتی ماژول \lr{AdaSplash Attention} با مهار تنک و هدایت انطباقی وزن‌های توجه'
    },
    'goldfinch_ahn': {
        'img': 'img/arch_goldfinch_ahn.jpeg',
        'caption': r'ساختار شبکه عصبی توجه ناهمگن \lr{Goldfinch AHN} و فشرده‌سازی پروجکشن‌های لایه'
    },
    'halo_regla': {
        'img': 'img/arch_regla_attention.jpeg',
        'caption': r'معماری توجه خطی گیت‌شده و منظم‌شده \lr{ReGLA (Regularized Gated Linear Attention)}'
    },
    'nsa_attention': {
        'img': 'img/arch_nsa_attention.jpeg',
        'caption': r'معماری توجه بومی تنک \lr{NSA (Native Sparse Attention)} با مسیریابی سلسله‌مراتبی توکن‌ها'
    },
    'nha_attention': {
        'img': 'img/arch_nha_attention.jpeg',
        'caption': r'ساختار توجه ناهمگن \lr{NHA (Non-Homogeneous Attention)} با جداسازی الگوهای محلی و سراسری'
    },
    'squeezing_heads_distiller': {
        'img': 'img/arch_squeezing_heads_distillation.jpeg',
        'caption': r'لایه تقطیر توجه و فشرده‌سازی چندسره \lr{(Squeezing Heads Attention Distillation Layer)}'
    },
    'progressive_token_pruning': {
        'img': 'img/arch_progressive_token_pruning.jpeg',
        'caption': r'جریان تانسوری هرس تدریجی توکن‌ها \lr{(Progressive Token Pruning)} در طول لایه‌های عمیق مدل'
    },
    'eat_attention': {
        'img': 'img/arch_eat_attention.jpeg',
        'caption': r'معماری ترنسفورمر توجه کارآمد \lr{EAT (Efficient Attention Transformer)}'
    },
    'infini_attention': {
        'img': 'img/arch_infini_attention.jpeg',
        'caption': r'معماری ماژول \lr{Infini-Attention} با تلفیق حافظه فشرده‌ساز پیوسته و پنجره توجه لغزان محلی'
    },
    'optimized_hybrid_attention': {
        'img': 'img/arch_optimized_hybrid_attention.jpeg',
        'caption': r'ساختار بلوکی توجه ترکیبی بهینه‌سازی‌شده \lr{(Optimized Hybrid Attention)}'
    },
    'nystrom_attention': {
        'img': 'img/arch_nystrom_attention.jpeg',
        'caption': r'معماری تفصیلی ماژول توجه نایستروم \lr{(Nyström Attention)} با استخراج نقاط راهنما \lr{(Landmark)} و معکوس‌سازی شولتز \lr{(Schulz Pinv)}'
    },
    'soa_fused_module': {
        'img': 'img/arch_shared_orthogonal_attention.jpeg',
        'caption': r'معماری توجه متعامد اشتراکی \lr{SOA (Shared Orthogonal Attention)} با تنگناهای نهان اختصاصی به ازای هر پروجکشن'
    },
    'dobi_attention': {
        'img': 'img/arch_dobi_svd.jpeg',
        'caption': r'معماری \lr{Dobi-SVD}: تجزیه مقادیر منفرد مشتق‌پذیر با گیتینگ اسپکترال یادگیرنده \lr{(Learnable Spectral Gating)}'
    },
    'mla_attention': {
        'img': 'img/arch_mla_deepseek_v2.jpeg',
        'caption': r'معماری توجه نهان چندسره \lr{MLA (Multi-Head Latent Attention)} در مدل \lr{DeepSeek-V2} با بردار نهان فشرده مشترک $c_{KV}$'
    },
    'compressed_mha_attention': {
        'img': 'img/arch_compressed_mha_joint_svd.jpeg',
        'caption': r'معماری توجه چندسره فشرده از طریق \lr{SVD} برش‌خورده مشترک \lr{(Joint Truncated SVD)} روی پروجکشن‌های \lr{QKV}'
    },
    'svd_compressed_attention': {
        'img': 'img/arch_compressed_mha_joint_svd.jpeg',
        'caption': r'معماری توجه چندسره فشرده از طریق \lr{SVD} برش‌خورده مشترک \lr{(Joint Truncated SVD)} روی پروجکشن‌های \lr{QKV}'
    }
}

# 3. Comparison chart resolver
comp_files = [f.replace('_comparison.png', '') for f in os.listdir(os.path.join(BASE_DIR, 'thesis_final/img/comparison')) if f.endswith('_comparison.png')]

def get_comp_chart(mod):
    candidates = [
        mod,
        mod.replace('_attention', ''),
        mod.replace('_attn', ''),
        mod.replace('_module', ''),
        f'{mod}_attention',
        f'{mod}_attn'
    ]
    if mod == 'soa_fused_module':
        candidates.append('soa_fused_attn')
    if mod == 'squeezing_heads_distiller':
        candidates.append('squeezing_heads_distill')
        candidates.append('squeezing_heads')
    if mod == 'dobi_attention':
        candidates.append('dobi_svd')
    if mod == 'retention':
        candidates.append('retention')

    for c in candidates:
        if c in comp_files:
            return f'img/comparison/{c}_comparison.png'
    for cf in comp_files:
        if mod in cf or cf in mod:
            return f'img/comparison/{cf}_comparison.png'
    return f'img/comparison/{mod}_comparison.png'

# 4. Write updated chapter_attentions.tex
chapter_lines = ["% Chapter 4: Monograph inputs for curated 100 unique modules\n"]
for m in curated_100:
    chapter_lines.append(f"\\input{{{MODULES_DIR}/{m}.tex}}\n")

with open(CHAPTER_FILE, "w", encoding="utf-8") as f:
    f.writelines(chapter_lines)
print(f"Updated {CHAPTER_FILE} with 100 unique module inputs.")

# 5. Process each of the 100 module files
for idx, mod in enumerate(curated_100, 1):
    fpath = os.path.join(MODULES_DIR, f"{mod}.tex")
    if not os.path.exists(fpath):
        print(f"ERROR: {fpath} does not exist!")
        continue
    
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    escaped_mod = mod.replace("_", r"\_")

    # A. Format elegant header with \clearpage, double rules, and numbering
    header_tex = f"""\\clearpage
\\phantomsection
\\vspace*{{0.2cm}}
\\noindent\\rule{{\\textwidth}}{{1.2pt}}\\vspace{{-0.2cm}}
\\subsection{{کالبدشکافی عمیق و تحلیل ساختاری ماژول شماره {idx}: \\lr{{{escaped_mod}}}}}
\\label{{sec:{mod}}}
\\vspace{{-0.2cm}}\\noindent\\rule{{\\textwidth}}{{0.5pt}}
\\vspace{{0.4cm}}"""

    # Replace top section/subsection
    content = re.sub(r'^(?:%[^\n]*\n)*\\clearpage\s*\\phantomsection[\s\S]*?\\subsection\{[^\n]+\}\s*\\label\{sec:[^}]+\}\s*\\vspace\{[^\n]+\}\s*\\noindent\\rule[^\n]+\s*\\vspace\{[^\n]+\}', '', content, flags=re.MULTILINE)
    content = re.sub(r'^(?:%[^\n]*\n)*(?:\\clearpage\s*)?\\subsection\{[^\n]+\}\s*\\label\{sec:[^}]+\}', '', content, flags=re.MULTILINE)
    content = header_tex + "\n" + content.lstrip()

    # B. Remove any Section 12 (References) if still lingering
    sec12_pos = content.find("بخش ۱۲:")
    if sec12_pos != -1:
        # cut off section 12
        content = content[:sec12_pos]
        # remove preceding subsubsection if needed
        last_sub = content.rfind(r"\subsubsection*{")
        if last_sub != -1 and "بخش ۱۲:" in content[last_sub:]:
            content = content[:last_sub]

    # C. Build clean Section 11 (Figures & Charts)
    comp_chart = get_comp_chart(mod)
    has_arch = mod in ARCH_MAP
    
    if has_arch:
        arch_info = ARCH_MAP[mod]
        sec11_tex = f"""\\subsubsection*{{بخش ۱۱: تصاویر و نمودارها \\lr{{(Figures \\& Charts)}}}}
در این بخش دیاگرام تفصیلی گردش داده و نمودار مقایسه عملکرد سه‌معیاره ماژول ارائه گردیده است. شکل \\ref{{fig:arch_{mod}}} معماری تانسوری و ساختار پروجکشن‌های لایه را به تصویر می‌کشد و شکل \\ref{{fig:comp_{mod}}} عملکرد آن را در مقایسه با مدل پایه نشان می‌دهد.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.90\\textwidth]{{{arch_info['img']}}}
\\caption{{{arch_info['caption']}}}
\\label{{fig:arch_{mod}}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{{{comp_chart}}}
\\caption{{نمودار ارزیابی مقایسه‌ای سه‌شاخصی دقت، اوج مصرف حافظه و توان عملیاتی مکانیزم \\lr{{{escaped_mod}}} در برابر خط مبنا}}
\\label{{fig:comp_{mod}}}
\\end{{figure}}
"""
    else:
        sec11_tex = f"""\\subsubsection*{{بخش ۱۱: تصاویر و نمودارها \\lr{{(Figures \\& Charts)}}}}
در این بخش نمودار مقایسه عملکرد تجربی ماژول با خط پایه در سه شاخص کلیدی دقت، اوج مصرف حافظه گرافیکی و توان عملیاتی استنتاج ارائه گردیده است (شکل \\ref{{fig:comp_{mod}}}).

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{{{comp_chart}}}
\\caption{{نمودار ارزیابی مقایسه‌ای سه‌شاخصی دقت، اوج مصرف حافظه و توان عملیاتی مکانیزم \\lr{{{escaped_mod}}} در برابر خط مبنا}}
\\label{{fig:comp_{mod}}}
\\end{{figure}}
"""

    # Replace existing section 11
    sec11_match = re.search(r'\\subsubsection\*\{بخش ۱۱:[^}]*\}', content)
    if sec11_match:
        idx_s11 = sec11_match.start()
        content = content[:idx_s11] + sec11_tex
    else:
        content = content.rstrip() + "\n\n" + sec11_tex

    # Clean up double newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Save
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

print("Successfully updated all 100 module files with beautiful headers, unique numbering, and full figures!")
