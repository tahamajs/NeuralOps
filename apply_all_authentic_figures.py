import os, re

BASE_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final"
IMG_DIR = os.path.join(BASE_DIR, "img")
COMP_DIR = os.path.join(IMG_DIR, "comparison")
ARCH_DIR = os.path.join(IMG_DIR, "architecture")
MODULES_DIR = os.path.join(BASE_DIR, "attentions_latex", "individual_modules")

with open(os.path.join(BASE_DIR, "chapter_attentions.tex"), "r", encoding="utf-8") as f:
    modules_in_chapter = [line.split("{")[1].split("}")[0].replace(".tex", "").split("/")[-1] for line in f if "\\input{" in line]

overrides_arch = {
    "SOAFusedAttention": "img/arch_shared_orthogonal_attention.jpeg",
    "soa_module": "img/arch_soa_module.jpeg",
    "spectra_v2": "img/arch_aegis_spectron_fp8.jpeg",
    "spectron": "img/arch_aegis_spectron_fp8.jpeg",
    "aegis_v5": "img/arch_aegis_hybrid.jpeg",
    "projection_factorized_attn": "img/arch_factorized_attention.jpeg",
    "svd_projection_attention": "img/arch_factorized_attention.jpeg",
    "token_compressed_attention": "img/arch_progressive_token_pruning.jpeg",
    "compressed_mha": "img/arch_compressed_mha_joint_svd.jpeg",
    "svd_compressed": "img/arch_svd_compressed.jpeg",
    "tt": "img/architecture/tt_architecture.png",
    "primal_attention": "img/arch_primal_attention.jpeg",
    "fusion_attn": "img/arch_fusion_attn.jpeg",
    "gqa_attention": "img/arch_gqa_attention.jpeg",
    "adaptive_sparse_attention": "img/arch_adaptive_sparse_attention.jpeg",
    "adaptive_span_attention": "img/arch_adaptive_span_attention.jpeg",
    "adasplash_attention": "img/arch_adasplash_attention.jpeg",
    "nsa_attention": "img/arch_nsa_attention.jpeg",
    "squeezing_heads_distiller": "img/arch_squeezing_heads_distiller.jpeg",
    "minimal_nova": "img/arch_minimal_nova.jpeg",
    "minimal_prism": "img/arch_minimal_prism.jpeg",
    "minimal_svd": "img/arch_minimal_svd.jpeg",
    "mla_attention": "img/arch_mla_attention.jpeg",
    "block_transformer_hiba": "img/arch_block_transformer_hiba.jpeg",
    "chela_slay": "img/arch_chela_slay.jpeg",
    "expert_choice_moe": "img/arch_expert_choice_moe.jpeg",
    "flash_attention3": "img/arch_flash_attention3.jpeg",
    "flash_attention3_fp8": "img/arch_flash_attention3_fp8.jpeg",
    "gated_delta_net": "img/arch_gated_delta_net.jpeg",
    "griffin_ssd": "img/arch_griffin_ssd.jpeg",
    "mamba": "img/arch_mamba.jpeg",
    "hgrn2_gated_deltanet": "img/arch_hgrn2_gated_deltanet_1.jpeg",
    "hgrn2_layer": "img/arch_hgrn2_layer_1.jpeg",
    "span_expanded_attention": "img/architecture/span_expanded_attention_architecture.png",
    "linformer_attention": "img/architecture/linformer_attention_architecture.png",
    "tpa_attention": "img/architecture/tpa_attention_architecture.png",
    "lora_attention": "img/architecture/lora_attention_architecture.png",
    "mixattention_moska": "img/architecture/mixattention_moska_architecture.png",
    "moe_deltakv_paged": "img/architecture/moe_deltakv_paged_architecture.png",
    "nasc_attention": "img/architecture/nasc_attention_architecture.png",
    "nexus": "img/architecture/nexus_architecture.png",
    "orion": "img/architecture/orion_architecture.png",
    "padre": "img/architecture/padre_architecture.png",
    "performer_attention": "img/architecture/performer_attention_architecture.png",
    "phantom": "img/architecture/phantom_architecture.png",
    "rala_attention": "img/architecture/rala_attention_architecture.png",
    "reap_moe": "img/architecture/reap_moe_architecture.png",
    "routing_attention": "img/architecture/routing_attention_architecture.png",
    "salt": "img/architecture/salt_architecture.png",
    "simple_ssm": "img/architecture/simple_ssm_architecture.png",
    "template_attention": "img/architecture/template_attention_architecture.png",
    "moba": "img/architecture/moba_architecture.png",
    "unified_rope": "img/architecture/unified_rope_architecture.png",
    "vspec": "img/architecture/vspec_architecture.png"
}

overrides_comp = {
    "soa_module": "img/comparison/soa_fused_module_comparison.png",
    "compressed_mha": "img/comparison/compressed_mha_attention_comparison.png",
    "tt": "img/comparison/tt_comparison.png",
    "SOAFusedAttention": "img/comparison/SOAFusedAttention_comparison.png",
}

def resolve_module_images(m):
    arch = overrides_arch.get(m)
    if not arch:
        stem = m.replace("_attention", "").replace("_attn", "")
        for cand in [f"arch_{m}_1.jpeg", f"arch_{stem}_1.jpeg", f"arch_{m}.jpeg", f"arch_{stem}.jpeg", f"{m}.jpeg", f"{stem}.jpeg", f"{m}.png"]:
            if os.path.exists(os.path.join(BASE_DIR, "img", cand)):
                arch = f"img/{cand}"
                break
        if not arch:
            arch = f"img/architecture/{m}_architecture.png"

    comp = overrides_comp.get(m)
    if not comp:
        stem = m.replace("_attention", "").replace("_attn", "")
        clean_m = m.replace("_module", "")
        for cand in [f"{m}_comparison.png", f"{stem}_comparison.png", f"{clean_m}_comparison.png", f"{m}_attention_comparison.png"]:
            if os.path.exists(os.path.join(COMP_DIR, cand)):
                comp = f"img/comparison/{cand}"
                break

    # Secondary arch
    sec = None
    stem = m.replace("_attention", "").replace("_attn", "")
    clean_m = m.replace("_module", "")
    for cand in [f"arch_{m}_2.jpeg", f"arch_{stem}_2.jpeg", f"{m}_2.jpeg", f"{stem}_2.jpeg"]:
        if os.path.exists(os.path.join(BASE_DIR, "img", cand)):
            sec = f"img/{cand}"
            break

    return arch, sec, comp

def process_module(mod):
    tex_path = os.path.join(MODULES_DIR, f"{mod}.tex")
    if not os.path.exists(tex_path):
        print(f"Skipping {mod}: file not found")
        return

    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    arch_img, sec_img, comp_img = resolve_module_images(mod)
    escaped_mod = mod.replace("_", r"\_")
    clean_mod = re.sub(r'[^a-zA-Z0-9]', '_', mod)

    # Determine header level (\subsection* vs \subsubsection*)
    sec_prefix = r"\subsection*"
    if r"\subsubsection*{بخش ۱:" in content or r"\subsubsection*{بخش 1:" in content:
        sec_prefix = r"\subsubsection*"

    # Construct the authentic Section 11 block
    sec11_block = f"{sec_prefix}{{بخش ۱۱: تصاویر و نمودارها \\lr{{(Figures \\& Charts)}}}}\n"
    sec11_block += f"در این بخش، بلوک‌دیاگرام جریان داده و معماری تفصیلی و نمودار ارزیابی مقایسه‌ای عملکرد مکانیزم \\lr{{{escaped_mod}}} ارائه شده است.\n\n"

    sec11_block += f"\\begin{{figure}}[H]\n\\centering\n"
    sec11_block += f"\\includegraphics[width=0.92\\textwidth]{{{arch_img}}}\n"
    sec11_block += f"\\caption{{دیاگرام ساختار معماری و جریان تانسوری ماژول \\lr{{{escaped_mod}}}}}\n"
    sec11_block += f"\\label{{fig:arch_{clean_mod}}}\n"
    sec11_block += f"\\end{{figure}}\n\n"

    if sec_img:
        sec11_block += f"\\begin{{figure}}[H]\n\\centering\n"
        sec11_block += f"\\includegraphics[width=0.92\\textwidth]{{{sec_img}}}\n"
        sec11_block += f"\\caption{{جزئیات تکمیلی پردازش و نگاشت تانسورها در ماژول \\lr{{{escaped_mod}}}}}\n"
        sec11_block += f"\\label{{fig:arch_sec_{clean_mod}}}\n"
        sec11_block += f"\\end{{figure}}\n\n"

    sec11_block += f"\\begin{{figure}}[H]\n\\centering\n"
    sec11_block += f"\\includegraphics[width=0.92\\textwidth]{{{comp_img}}}\n"
    sec11_block += f"\\caption{{نمودار ارزیابی مقایسه‌ای سه‌شاخصی دقت، اوج مصرف حافظه و توان عملیاتی مکانیزم \\lr{{{escaped_mod}}} در برابر خط مبنا}}\n"
    sec11_block += f"\\label{{fig:comp_{clean_mod}}}\n"
    sec11_block += f"\\end{{figure}}\n"

    # 1. Remove all \fbox figure blocks everywhere in the content
    content = re.sub(r'\\begin\{figure\}[^}]*?\\fbox\{[\s\S]*?\\end\{figure\}', '', content)
    # Remove standalone fbox if any remains
    content = re.sub(r'\\fbox\{\\parbox\{[^\}]+\}\{[\s\S]*?\}\}', '', content)

    # 2. Check if Section 12 (References) is present
    sec12_match = re.search(r'(\\subsubsection\*\{بخش\s*۱۲:[^\}]+\}[\s\S]*?$|\\subsection\*\{بخش\s*۱۲:[^\}]+\}[\s\S]*?$)', content)
    sec12_block = ""
    if sec12_match:
        sec12_block = "\n\n" + sec12_match.group(1).strip()
        # Remove Section 12 from content before processing Section 11
        content = content[:sec12_match.start()].rstrip()

    # 3. Check if Section 11 already exists in content
    sec11_match = re.search(r'(\\subsubsection\*\{بخش\s*۱۱:[^\}]+\}[\s\S]*?$|\\subsection\*\{بخش\s*۱۱:[^\}]+\}[\s\S]*?$)', content)
    if sec11_match:
        # Replace Section 11 directly
        content = content[:sec11_match.start()].rstrip() + "\n\n" + sec11_block + sec12_block
    else:
        # Also check if there are figures at the very end of the file that were previous standalone figures
        content = re.sub(r'\\begin\{figure\}[\s\S]*?\\end\{figure\}\s*$', '', content).rstrip()
        # Append Section 11 + Section 12
        content = content + "\n\n" + sec11_block + sec12_block

    # Final cleanup of excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content).strip() + "\n"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Updated {mod}.tex successfully.")

for idx, m in enumerate(modules_in_chapter, 1):
    process_module(m)

print(f"\nAll {len(modules_in_chapter)} modules updated with authentic Section 11 images!")
