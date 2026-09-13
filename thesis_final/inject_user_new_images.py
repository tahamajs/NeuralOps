import os
import shutil
import re

IMG_DIR = "thesis_final/img"
MODULES_DIR = "thesis_final/attentions_latex/individual_modules"

# Mapping from original filename to (clean_filename, module_name)
NEW_IMAGES_MAP = {
    "AEGIS-Attention V1.jpeg": ("arch_aegis_v1.jpeg", "aegis_v1"),
    "ASVD Attention.jpeg": ("arch_asvd_attention.jpeg", "asvd_attention"),
    "Adaptive SVD Attention.jpeg": ("arch_adaptive_svd_attention.jpeg", "adaptive_svd_attention"),
    "Aegis-Spectron-FP8.jpeg": ("arch_aegis_spectron_fp8.jpeg", "aegis_spectron_fp8"),
    "BAM-UMoE.jpeg": ("arch_bam_umoe.jpeg", "bam_umoe"),
    "Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).jpeg": ("arch_block_transformer_hiba.jpeg", "block_transformer_hiba"),
    "CHELA-SLAY.jpeg": ("arch_chela_slay.jpeg", "chela_slay"),
    "ClusterFormer.jpeg": ("arch_cluster_attention.jpeg", "cluster_attention"),
    "Cosformer.jpeg": ("arch_cosformer.jpeg", "cosformer"),
    "Decay Linear Attention.jpeg": ("arch_decay_linear_attention.jpeg", "decay_linear_attention"),
    "Expert Choice Mixture of Experts (MoE) Attention.jpeg": ("arch_expert_choice_moe.jpeg", "expert_choice_moe"),
    "FlashAttention-3 FP8.jpeg": ("arch_flash_attention3_fp8.jpeg", "flash_attention3_fp8"),
    "FlashAttention-3.jpeg": ("arch_flash_attention3.jpeg", "flash_attention3"),
    "Galerkin Attention.jpeg": ("arch_galerkin_attention.jpeg", "galerkin_attention"),
    "Gated DeltaNet Block with Optional Multi-Head Attention Bypass.jpeg": ("arch_gated_deltanet_block.jpeg", "gated_deltanet_block"),
    "Gated DeltaNet- Delta Rule Linear Recurrent Attention.jpeg": ("arch_gated_delta_net.jpeg", "gated_delta_net"),
    "Gated Linear Attention (GLA).jpeg": ("arch_gated_linear_attention.jpeg", "gated_linear_attention"),
    "Griffin SSD.jpeg": ("arch_griffin_ssd.jpeg", "griffin_ssd"),
    "HymbaTransX.jpeg": ("arch_hymba_transx.jpeg", "hymba_transx"),
    "Information-Bottlenecked Jamba.jpeg": ("arch_jamba_info.jpeg", "jamba_info"),
    "Kernelized Decay Attention (KDA).jpeg": ("arch_kda_attention.jpeg", "kda_attention"),
    "LEKAv1.jpeg": ("arch_leka_v1.jpeg", "leka_v1"),
    "LEKAv2.jpeg": ("arch_leka_v2.jpeg", "leka_v2"),
    "LEKAv3.jpeg": ("arch_leka_v3.jpeg", "leka_v3"),
}

def setup_images():
    for orig_name, (clean_name, mod) in NEW_IMAGES_MAP.items():
        src = os.path.join(IMG_DIR, orig_name)
        dst = os.path.join(IMG_DIR, clean_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"Copied {orig_name} -> {clean_name}")
        else:
            print(f"Source image missing: {src}")

def update_module_tex():
    updated = 0
    for orig_name, (clean_name, mod) in NEW_IMAGES_MAP.items():
        tex_path = os.path.join(MODULES_DIR, f"{mod}.tex")
        if not os.path.exists(tex_path):
            print(f"Tex file not found: {tex_path}")
            continue

        with open(tex_path, "r", encoding="utf-8") as f:
            c = f.read()

        new_img_rel = f"img/{clean_name}"
        
        # Replace the architecture image in Section 11 or wherever \includegraphics is used
        # Typically looks like: \includegraphics[width=0.95\textwidth]{img/architecture/<mod>_architecture.png}
        # Or \includegraphics[...]{img/...}
        pattern = rf"(\\includegraphics\[[^\]]*\]\{{)img/architecture/{mod}_architecture\.png(\}})"
        if re.search(pattern, c):
            new_c = re.sub(pattern, rf"\1{new_img_rel}\2", c)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(new_c)
            updated += 1
            print(f"Updated {mod}.tex -> {new_img_rel}")
        else:
            # Check if there is already an arch image or another pattern
            arch_pattern = rf"(\\includegraphics\[[^\]]*\]\{{)(img/[^}}]+arch[^}}]*|img/[^}}]+)(\}})"
            # Let's check first figure
            fig_match = re.search(r"\\begin\{figure\}.*?\\label\{fig:arch_" + mod + r"\}.*?\\end\{figure\}", c, re.DOTALL)
            if fig_match:
                fig_block = fig_match.group(0)
                new_fig_block = re.sub(r"(\\includegraphics\[[^\]]*\]\{)[^}]+(\})", rf"\1{new_img_rel}\2", fig_block, count=1)
                new_c = c.replace(fig_block, new_fig_block)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(new_c)
                updated += 1
                print(f"Updated (via fig block) {mod}.tex -> {new_img_rel}")
            else:
                print(f"Could not find arch figure pattern in {mod}.tex")

    print(f"Total modules updated with new user images: {updated}")

if __name__ == "__main__":
    setup_images()
    update_module_tex()
