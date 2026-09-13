import os, re

kept_90 = [
    'full_attention', 'fusion_attn', 'primal_attention', 'gqa_attention',
    'adaptive_sparse_attention', 'adaptive_span_attention', 'adasplash_attention',
    'goldfinch_ahn', 'halo_regla', 'nsa_attention', 'nha_attention',
    'squeezing_heads_distiller', 'progressive_token_pruning', 'eat_attention',
    'infini_attention', 'optimized_hybrid_attention', 'asvd_attention',
    'rpc_attention', 'jamba', 'svd_mamba',
    'compressed_mha_attention', 'svd_linear_attention', 'res_svd',
    'minimal_aurora', 'minimal_nova', 'minimal_prism', 'minimal_svd',
    'aegis_hybrid_attention', 'svd_llm_v2_attention', 'svd_compressed_attention',
    'mla_attention', 'dobi_attention', 'factorized_attention', 'soa_fused_module',
    'nystrom_attention', 'hla_attention', 'hamming_attention', 'hgrn2_gated_deltanet',
    'hgrn2_layer', 'SOAFusedAttention', 'adaptive_svd_attention', 'aegis_spectron_fp8',
    'aegis_v1', 'bam_umoe', 'block_transformer_hiba', 'chela_slay', 'cluster_attention',
    'cosformer', 'decay_linear_attention', 'expert_choice_moe', 'flash_attention3',
    'flash_attention3_fp8', 'galerkin_attention', 'gated_delta_net',
    'gated_deltanet_block', 'gated_linear_attention', 'griffin_ssd', 'hymba_transx',
    'jamba_info', 'kda_attention', 'leka_v1', 'leka_v2', 'leka_v3', 'linformer_attention',
    'lola_attention', 'lora_attention', 'lsh_attention', 'mamba',
    'mixattention_moska', 'moba_attention', 'moe_deltakv_paged', 'nasc_attention',
    'nexus', 'orion', 'padre', 'performer_attention', 'phantom',
    'projection_factorized_attn', 'rala_attention', 'reap_moe', 'regla_attention',
    'routing_attention', 'salt', 'simple_ssm',
    'span_expanded_attention', 'spectra_v2_attention',
    'svd_projection_attention', 'svd_sequence_attention', 'template_attention',
    'token_compressed_attention'
]

print(f'Total kept pure-passing modules: {len(kept_90)}')

img_dir = 'img'
arch_dir = 'img/architecture'
comp_dir = 'img/comparison'

# 1. Update chapter_attentions.tex
tex_lines = ['% Chapter 4: Monograph inputs for curated 90 pure-passing successful modules (Zero fails, zero zeros)\n']
for m in kept_90:
    tex_lines.append(f'\\input{{/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules/{m}.tex}}\n')

with open('chapter_attentions.tex', 'w', encoding='utf-8') as f:
    f.writelines(tex_lines)
print('chapter_attentions.tex updated.')

# 2. Process all 90 files
for idx, mod in enumerate(kept_90, 1):
    stem = mod.replace('_attention', '')
    clean_mod = re.sub(r'[^a-zA-Z0-9]', '_', mod)
    escaped_mod = mod.replace('_', r'\_')

    # Comparison chart
    comp_img = None
    for cp in [f'{comp_dir}/{mod}_comparison.png', f'{comp_dir}/{stem}_comparison.png', f'{comp_dir}/{mod.replace("_module", "")}_comparison.png']:
        if os.path.exists(cp):
            comp_img = cp
            break

    # Architecture diagram
    primary_arch = None
    secondary_arch = None

    custom_cands = [
        f'{img_dir}/arch_{mod}.jpeg', f'{img_dir}/arch_{stem}.jpeg',
        f'{img_dir}/arch_{mod}_1.jpeg', f'{img_dir}/arch_{stem}_1.jpeg',
        f'{img_dir}/arch_{mod}.jpg', f'{img_dir}/arch_{stem}.jpg',
        f'{img_dir}/arch_{mod}.png', f'{img_dir}/arch_{stem}.png'
    ]
    for c in custom_cands:
        if os.path.exists(c):
            primary_arch = c
            break

    sec_cands = [
        f'{img_dir}/arch_{mod}_2.jpeg', f'{img_dir}/arch_{stem}_2.jpeg',
        f'{img_dir}/arch_{mod}_2.png'
    ]
    for sc in sec_cands:
        if os.path.exists(sc):
            secondary_arch = sc
            break

    if not primary_arch:
        arch_cands = [
            f'{arch_dir}/{mod}_architecture.png',
            f'{arch_dir}/{stem}_architecture.png',
            f'{arch_dir}/{mod}_attn_architecture.png',
            f'{arch_dir}/{stem}_attn_architecture.png'
        ]
        for ac in arch_cands:
            if os.path.exists(ac):
                primary_arch = ac
                break

    fpath = f'attentions_latex/individual_modules/{mod}.tex'
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find where the genuine content begins using multiple anchors
    anchors = ['بخش ۱: شناسنامه و معرفی', '۱. فلسفه طراحی', 'فلسفه طراحی', 'مقدمه و شناسنامه']
    positions = [content.find(a) for a in anchors if content.find(a) != -1]
    if positions:
        first_pos = min(positions)
        sec1_start = content.rfind('\\', 0, first_pos)
        body = content[sec1_start:]
    else:
        body = content

    # Cut off existing Section 11 from body if present
    sec11_idx = body.find(r'\subsection*{بخش ۱۱: تصاویر و نمودارها')
    if sec11_idx == -1:
        sec11_idx = body.find(r'\subsection{بخش ۱۱: تصاویر و نمودارها')
    if sec11_idx != -1:
        body = body[:sec11_idx].rstrip()

    banner = (
        f'\\clearpage\n'
        f'\\phantomsection\n'
        f'\\vspace*{{0.2cm}}\n'
        f'\\noindent\\rule{{\\textwidth}}{{1.2pt}}\n'
        f'\\vspace*{{0.1cm}}\n'
        f'\\subsection{{کالبدشکافی عمیق و تحلیل ساختاری ماژول شماره {idx}: \\lr{{{escaped_mod}}}}}\n'
        f'\\label{{sec:mod_{clean_mod}}}\n'
        f'\\vspace*{{-0.2cm}}\n'
        f'\\noindent\\rule{{\\textwidth}}{{0.5pt}}\n'
        f'\\vspace*{{0.3cm}}\n\n'
    )

    sec11 = f'\\subsection*{{بخش ۱۱: تصاویر و نمودارها (Figures \\& Charts)}}\n\n'
    sec11 += f'در این بخش، دیاگرام تفصیلی گردش داده و نمودارهای عملکردی ماژول \\lr{{{escaped_mod}}} ارائه شده است.\n\n'

    if primary_arch:
        sec11 += (
            f'\\begin{{figure}}[htbp]\n'
            f'\\centering\n'
            f'\\includegraphics[width=0.95\\textwidth]{{{primary_arch}}}\n'
            f'\\caption{{دیاگرام ساختار معماری و جریان تانسوری ماژول \\lr{{{escaped_mod}}}}}\n'
            f'\\label{{fig:arch_{clean_mod}}}\n'
            f'\\end{{figure}}\n\n'
        )
    if secondary_arch:
        sec11 += (
            f'\\begin{{figure}}[htbp]\n'
            f'\\centering\n'
            f'\\includegraphics[width=0.95\\textwidth]{{{secondary_arch}}}\n'
            f'\\caption{{جزئیات تکمیلی پردازش و نگاشت تانسورها در ماژول \\lr{{{escaped_mod}}}}}\n'
            f'\\label{{fig:arch_sec_{clean_mod}}}\n'
            f'\\end{{figure}}\n\n'
        )
    if comp_img:
        sec11 += (
            f'\\begin{{figure}}[htbp]\n'
            f'\\centering\n'
            f'\\includegraphics[width=0.95\\textwidth]{{{comp_img}}}\n'
            f'\\caption{{نمودار ارزیابی مقایسه‌ای سه‌شاخصی دقت، اوج مصرف حافظه و توان عملیاتی مکانیزم \\lr{{{escaped_mod}}} در برابر خط مبنا}}\n'
            f'\\label{{fig:comp_{clean_mod}}}\n'
            f'\\end{{figure}}\n\n'
        )

    new_content = banner + body + '\n\n' + sec11

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_content)

print('Successfully processed all 90 modules with clean banners and images!')
