import os, re

MODULES_DIR = 'thesis_final/attentions_latex/individual_modules'

# Mapping of module files to their respective image, caption, and label
MAPPINGS = [
    {
        'files': ['full_attention.tex'],
        'img': 'img/arch_full_attention.jpeg',
        'caption': r'نمودار گردش داده و معماری تانسوری مکانیزم توجه چندسره استاندارد \lr{(Full Multi-Head Attention)} به همراه انطباق \lr{LoRA} و پروجکشن‌های خطی',
        'label': 'fig:full_attention_arch'
    },
    {
        'files': ['fusion_attn.tex'],
        'img': 'img/arch_fusion_attn.jpeg',
        'caption': r'معماری هیبریدی ماژول \lr{FusionAttention} با تفکیک مسیر سریع برای توالی‌های کوتاه \lr{($N \le 128$)} و توجه تنک بلوکی \lr{(Top-K Block Selection)} برای توالی‌های بلند',
        'label': 'fig:fusion_attn_arch'
    },
    {
        'files': ['primal_attention.tex'],
        'img': 'img/arch_primal_attention.jpeg',
        'caption': r'نمودار ساختار تانسوری ماژول توجه بدوی \lr{(Primal Attention)} و فاکتورسازی فضای ویژگی‌ها',
        'label': 'fig:primal_attention_arch'
    },
    {
        'files': ['gqa_attention.tex', 'wgqa_attention.tex', 'wgqa.tex'],
        'img': 'img/arch_grouped_query_attention.jpeg',
        'caption': r'معماری توجه با پرس‌وجوی گروه‌بندی‌شده \lr{(Grouped-Query Attention - GQA)} جهت بهینه‌سازی حافظه \lr{KV-Cache}',
        'label': 'fig:gqa_attention_arch'
    },
    {
        'files': ['adaptive_sparse_attention.tex'],
        'img': 'img/arch_adaptive_sparse_attention.jpeg',
        'caption': r'معماری تانسوری توجه تنک تطبیقی \lr{(Adaptive Sparse Attention)} با گزینش پویای الگوهای توجه',
        'label': 'fig:adaptive_sparse_attention_arch'
    },
    {
        'files': ['adaptive_span_attention.tex'],
        'img': 'img/arch_adaptive_span_attention.jpeg',
        'caption': r'سازوکار تنظیم پویای طول پنجره زمینه در ماژول \lr{Adaptive Span Attention}',
        'label': 'fig:adaptive_span_attention_arch'
    },
    {
        'files': ['adasplash_attention.tex'],
        'img': 'img/arch_adasplash_attention.jpeg',
        'caption': r'معماری محاسباتی ماژول \lr{AdaSplash Attention} با مهار تنک و هدایت انطباقی وزن‌های توجه',
        'label': 'fig:adasplash_attention_arch'
    },
    {
        'files': ['goldfinch_ahn.tex'],
        'img': 'img/arch_goldfinch_ahn.jpeg',
        'caption': r'ساختار شبکه عصبی توجه ناهمگن \lr{Goldfinch AHN} و فشرده‌سازی پروجکشن‌های لایه',
        'label': 'fig:goldfinch_ahn_arch'
    },
    {
        'files': ['halo_regla.tex', 'regla_attention.tex'],
        'img': 'img/arch_regla_attention.jpeg',
        'caption': r'معماری توجه خطی گیت‌شده و منظم‌شده \lr{ReGLA (Regularized Gated Linear Attention)}',
        'label': 'fig:regla_attention_arch'
    },
    {
        'files': ['nsa_attention.tex'],
        'img': 'img/arch_nsa_attention.jpeg',
        'caption': r'معماری توجه بومی تنک \lr{NSA (Native Sparse Attention)} با مسیریابی سلسله‌مراتبی توکن‌ها',
        'label': 'fig:nsa_attention_arch'
    },
    {
        'files': ['nha_attention.tex', 'nha_compressor.tex'],
        'img': 'img/arch_nha_attention.jpeg',
        'caption': r'ساختار توجه ناهمگن \lr{NHA (Non-Homogeneous Attention)} با جداسازی الگوهای محلی و سراسری',
        'label': 'fig:nha_attention_arch'
    },
    {
        'files': ['squeezing_heads_distiller.tex', 'squeezing_heads.tex', 'squeezing_heads_distill.tex'],
        'img': 'img/arch_squeezing_heads_distillation.jpeg',
        'caption': r'لایه تقطیر توجه و فشرده‌سازی چندسره \lr{(Squeezing Heads Attention Distillation Layer)}',
        'label': 'fig:squeezing_heads_arch'
    },
    {
        'files': ['progressive_token_pruning.tex'],
        'img': 'img/arch_progressive_token_pruning.jpeg',
        'caption': r'جریان تانسوری هرس تدریجی توکن‌ها \lr{(Progressive Token Pruning)} در طول لایه‌های عمیق مدل',
        'label': 'fig:progressive_token_pruning_arch'
    },
    {
        'files': ['eat_attention.tex', 'eat.tex'],
        'img': 'img/arch_eat_attention.jpeg',
        'caption': r'معماری ترنسفورمر توجه کارآمد \lr{EAT (Efficient Attention Transformer)}',
        'label': 'fig:eat_attention_arch'
    },
    {
        'files': ['infini_attention.tex'],
        'img': 'img/arch_infini_attention.jpeg',
        'caption': r'معماری ماژول \lr{Infini-Attention} با تلفیق حافظه فشرده‌ساز پیوسته و پنجره توجه لغزان محلی',
        'label': 'fig:infini_attention_arch'
    },
    {
        'files': ['optimized_hybrid_attention.tex'],
        'img': 'img/arch_optimized_hybrid_attention.jpeg',
        'caption': r'ساختار بلوکی توجه ترکیبی بهینه‌سازی‌شده \lr{(Optimized Hybrid Attention)}',
        'label': 'fig:optimized_hybrid_attention_arch'
    },
    {
        'files': ['nystrom_attention.tex'],
        'img': 'img/arch_nystrom_attention.jpeg',
        'caption': r'معماری تفصیلی ماژول توجه نایستروم \lr{(Nyström Attention)} با استخراج نقاط راهنما \lr{(Landmark)} و معکوس‌سازی شولتز \lr{(Schulz Pinv)}',
        'label': 'fig:nystrom_attention_arch'
    },
    {
        'files': ['soa_attn.tex', 'soa_module.tex', 'soa_fused_module.tex', 'SOAFusedAttention.tex'],
        'img': 'img/arch_shared_orthogonal_attention.jpeg',
        'caption': r'معماری توجه متعامد اشتراکی \lr{SOA (Shared Orthogonal Attention)} با تنگناهای نهان اختصاصی به ازای هر پروجکشن',
        'label': 'fig:shared_orthogonal_attention_arch'
    },
    {
        'files': ['dobi_attention.tex'],
        'img': 'img/arch_dobi_svd.jpeg',
        'caption': r'معماری \lr{Dobi-SVD}: تجزیه مقادیر منفرد مشتق‌پذیر با گیتینگ اسپکترال یادگیرنده \lr{(Learnable Spectral Gating)}',
        'label': 'fig:dobi_svd_arch'
    },
    {
        'files': ['mla_attention.tex'],
        'img': 'img/arch_mla_deepseek_v2.jpeg',
        'caption': r'معماری توجه نهان چندسره \lr{MLA (Multi-Head Latent Attention)} در مدل \lr{DeepSeek-V2} با بردار نهان فشرده مشترک $c_{KV}$',
        'label': 'fig:mla_attention_arch'
    },
    {
        'files': ['compressed_mha.tex', 'compressed_mha_attention.tex', 'svd_compressed.tex', 'svd_compressed_attention.tex'],
        'img': 'img/arch_compressed_mha_joint_svd.jpeg',
        'caption': r'معماری توجه چندسره فشرده از طریق \lr{SVD} برش‌خورده مشترک \lr{(Joint Truncated SVD)} روی پروجکشن‌های \lr{QKV}',
        'label': 'fig:compressed_mha_joint_svd_arch'
    }
]

def make_figure_tex(img_path, caption, label, mod_name):
    unique_label = f"{label}_{mod_name.replace('.', '_')}"
    return f"""\\subsubsection*{{بخش ۱۱: تصاویر و نمودارها \\lr{{(Figures \\& Charts)}}}}
در این بخش دیاگرام تفصیلی گردش داده و ساختار معماری ماژول ارائه گردیده است. شکل \\ref{{{unique_label}}} ساختار تانسوری و اجزای درونی این مکانیزم را به نمایش می‌گذارد.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{{{img_path}}}
\\caption{{{caption}}}
\\label{{{unique_label}}}
\\end{{figure}}
"""

for item in MAPPINGS:
    for fname in item['files']:
        fpath = os.path.join(MODULES_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Skipping {fname} (not found)")
            continue
        
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        mod_name = os.path.splitext(fname)[0]
        fig_tex = make_figure_tex(item['img'], item['caption'], item['label'], mod_name)
        
        # Check if already has this image
        if item['img'] in content:
            print(f"Already updated: {fname}")
            continue
            
        # Find section 11
        sec11_match = re.search(r'\\subsubsection\*\{بخش ۱۱:[^}]*\}', content)
        if sec11_match:
            # Replace from sec11 to end of file
            idx = sec11_match.start()
            new_content = content[:idx] + fig_tex
        else:
            # Append to the end
            new_content = content.rstrip() + "\n\n" + fig_tex
            
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully inserted figure into {fname}")

