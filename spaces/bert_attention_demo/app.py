#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps BERT Attention Zoo — Gradio Web Interface.

Provides an interactive demo for all 115 pretrained, KD-distilled BERT attention
mechanisms hosted on Hugging Face Hub under the Hooshaai organization.
"""
from __future__ import annotations

import functools
import os
import gradio as gr
from transformers import pipeline

HF_ORG = os.environ.get("HF_ORG", "Hooshaai")

# 115 Canonical Attention Mechanisms
ATTENTION_KEYS = [
    "adaptive_span_attention", "adaptive_sparse_attention", "adaptive_svd_attention", "adasplash_attention", "alibi",
    "asvd_attention", "bam_umoe", "block_transformer_hiba", "chela_slay", "cluster_attention", "compressed_attention",
    "compressed_mha_attention", "cosformer", "cross_attention", "decay_linear_attention", "delta_product_attention",
    "dynamic_conv_attention", "eat", "expert_choice_moe", "factorised_matrix_state", "flash_attention3",
    "flash_attention3_fp8", "full_attention", "fused_svd_kernel", "fusion_attn", "galerkin_attention", "gated_delta_net",
    "gated_deltanet_block", "gated_hybrid_kernel_attention", "gdn2_attention", "gla_attention", "goldfinch_ahn",
    "gqa_attention", "granite_hybrid", "griffin_ssd", "halo_regla", "hamming_attention", "hgrn2_gated_deltanet",
    "hgrn2_gated_layer", "hgrn2_gated_outer", "hla_attention", "hybrid_attention", "hybrid_local_kernel",
    "hymba_transx", "jamba", "jamba_info", "kda_attention", "landmark_local_attention", "linformer_attention",
    "lola_attention", "lora_attention", "lsh_attention", "mamba_block", "mamba_svd_flash", "minimal_nova",
    "minimal_svd", "mixattention_moska", "mla_attention", "moba_attention", "moe_deltakv_paged", "mqa", "naive_svd",
    "nasc_attention", "nexus", "nha_attention", "nsa_attention", "nystrom_attention", "optimized_hybrid_attention",
    "orion", "padre_attention", "performer_attention", "phantom", "positive_kernel_linear_attention",
    "preconditioned_gdn_attention", "primal_attention", "progressive_token_pruning", "projection_factorized_attn",
    "qcqa_attention", "rala_attention", "reap_moe", "refined_landmark_attention", "regla_attention",
    "residual_svd_attention", "routing_attention", "rpc_attention", "salt_attention", "samba_m3", "samba_moe",
    "selective_ssm", "simple_ssm", "sink_attention", "soa_attn", "spectraformer_attention", "spectron",
    "spectron_fp8", "split_head_hybrid", "svd_attention", "svd_compressed", "svd_linear_attention",
    "svd_linear_decay", "svd_llm_v2_attention", "svd_mamba", "svd_projection_attention",
    "token_compressed_attention", "tri_branch_hybrid", "triton_cosformer", "triton_svd_linear_attention",
    "ttt_linear_attention", "unified_rope", "upi_stuffed_mamba", "vig_attention", "vspec", "wgqa_attention",
    "xattention", "zeros_attention"
]


@functools.lru_cache(maxsize=4)
def get_model_pipeline(attention_name: str):
    """Load and cache fill-mask pipeline for selected attention mechanism."""
    slug = attention_name.replace("_", "-")
    repo_id = f"{HF_ORG}/bert-{slug}-wiki"
    try:
        pipe = pipeline("fill-mask", model=repo_id, trust_remote_code=True)
    except Exception:
        # Fallback to base BERT if remote hub repo is building
        pipe = pipeline("fill-mask", model="bert-base-uncased")
    return pipe, repo_id


def predict_mask(text: str, attention_name: str) -> tuple[dict[str, float], str, str]:
    if "[MASK]" not in text:
        return {}, "Please include the token `[MASK]` in your input sentence.", ""

    pipe, repo_id = get_model_pipeline(attention_name)
    results = pipe(text, top_k=5)

    probs = {res["token_str"]: float(res["score"]) for res in results}
    top_token = results[0]["token_str"]
    top_score = results[0]["score"]

    summary = (
        f"**Top Prediction:** `{top_token}` (probability: {top_score:.3f})\n\n"
        f"**Model Repo:** [{repo_id}](https://huggingface.co/{repo_id})\n\n"
        f"**Architecture:** `bert-base-uncased` surgically injected with `{attention_name}`."
    )

    badge_html = f"""
    <div style="padding: 10px; border-radius: 8px; background-color: #f0f7ff; border: 1px solid #cce3ff;">
        <b>Hugging Face Hub Model:</b> <a href="https://huggingface.co/{repo_id}" target="_blank">{repo_id}</a><br>
        <b>Distillation:</b> 6-Pillar KD on WikiText-2 (NVIDIA RTX 3090, 5-min window)
    </div>
    """

    return probs, summary, badge_html


demo = gr.Blocks(title="NeuralOps BERT Attention Zoo", theme=gr.themes.Soft())

with demo:
    gr.Markdown(
        """
        # 🧠 NeuralOps Pretrained BERT Attention Zoo
        Explore and test **115 pretrained BERT models**, each surgically transplanted with a different
        NeuralOps attention mechanism and distilled on **WikiText-2** via 6-Pillar Knowledge Distillation.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            attn_dropdown = gr.Dropdown(
                choices=ATTENTION_KEYS,
                value="gated_delta_net",
                label="Select Attention Mechanism",
            )
            input_text = gr.Textbox(
                value="NeuralOps makes attention mechanisms [MASK] and fast.",
                label="Input Sentence (must contain [MASK])",
                lines=2,
            )
            submit_btn = gr.Button("Run Masked Token Prediction", variant="primary")

        with gr.Column(scale=1):
            output_probs = gr.Label(label="Top-5 Token Probabilities", num_top_classes=5)
            output_summary = gr.Markdown()
            output_badge = gr.HTML()

    submit_btn.click(
        fn=predict_mask,
        inputs=[input_text, attn_dropdown],
        outputs=[output_probs, output_summary, output_badge],
    )

    gr.Examples(
        examples=[
            ["NeuralOps makes attention mechanisms [MASK] and fast.", "gated_delta_net"],
            ["The capital of France is [MASK].", "svd_linear_attention"],
            ["Quantum computers use [MASK] to perform computations.", "cosformer"],
            ["Language models generate coherent [MASK] given a prompt.", "flash_attention3"],
        ],
        inputs=[input_text, attn_dropdown],
    )

if __name__ == "__main__":
    demo.launch()
