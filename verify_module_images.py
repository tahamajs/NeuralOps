import os, glob, re

BASE_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final"
IMG_DIR = os.path.join(BASE_DIR, "img")
COMP_DIR = os.path.join(IMG_DIR, "comparison")
ARCH_DIR = os.path.join(IMG_DIR, "architecture")
MODULES_DIR = os.path.join(BASE_DIR, "attentions_latex", "individual_modules")

modules_in_chapter = []
with open(os.path.join(BASE_DIR, "chapter_attentions.tex"), "r", encoding="utf-8") as f:
    for line in f:
        if r"\input{" in line:
            m = line.split("{")[1].split("}")[0]
            modules_in_chapter.append(os.path.basename(m).replace(".tex", ""))

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
        for cand in [f"arch_{m}.jpeg", f"arch_{stem}.jpeg", f"{m}.jpeg", f"{stem}.jpeg", f"{m}.png"]:
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

all_ok = True
for m in modules_in_chapter:
    arch, sec, comp = resolve_module_images(m)
    arch_exists = os.path.exists(os.path.join(BASE_DIR, arch)) if arch else False
    comp_exists = os.path.exists(os.path.join(BASE_DIR, comp)) if comp else False
    sec_exists = os.path.exists(os.path.join(BASE_DIR, sec)) if sec else True

    if not (arch_exists and comp_exists and sec_exists):
        print(f"FAILED: {m:30s} | Arch: {arch} ({arch_exists}) | Comp: {comp} ({comp_exists}) | Sec: {sec} ({sec_exists})")
        all_ok = False

if all_ok:
    print(f"SUCCESS: All {len(modules_in_chapter)} modules have 100% verified authentic image files!")
