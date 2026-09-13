import json

modules = [
    "padre", "performer_attention", "phantom", "projection_factorized_attn", 
    "rala_attention", "reap_moe", "regla_attention", "routing_attention", 
    "salt", "simple_ssm", "span_expanded_attention", "svd_projection_attention", 
    "template_attention", "token_compressed_attention", "moba", "spectra_v2", 
    "compressed_mha", "svd_compressed", "spectron", "soa_module", "aegis_v5"
]

subagents = []
for mod in modules:
    subagents.append({
        "TypeName": "research",
        "Role": f"Writer for {mod}",
        "Model": "pro",
        "Prompt": f"""You are an expert scientific researcher and Latex writer.
Your task is to completely rewrite the module documentation for {mod} in Latex.
Target File: /Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules/{mod}.tex
Code Base: /Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/ and Codes/framework/compression_engine/
Docs: /Users/tahamajs/Documents/uni/Project/Docs/content/attentions/
Master Results Source of Truth: /Users/tahamajs/Documents/uni/Project/master_verified_results.json

Instructions:
1. Search and read the genuine metrics for `{mod}` from master_verified_results.json.
2. Search and read the source code and docs for the module.
3. Rewrite the corresponding .tex file (create it if it doesn't exist) with maximum academic rigor, high mathematical depth, and exhaustive technical completeness.
Structure it with the 7 sections as specified:
   \section{{تحلیل علمی و کالبدشکافی ساختاری ماژول \lr{{{mod}}}}}
   \subsection*{{بخش ۱: ایده محوری و چرایی معماری (Architectural Rationale)}}: No clichés; deep theoretical rationale.
   \subsection*{{بخش ۲: فرمول‌بندی ریاضی و مبانی نظری (Mathematical Formulation)}}: Strict equations, all tensors/variables explicitly defined, time/space complexity O(...).
   \subsection*{{بخش ۳: کالبدشکافی سورس‌کد و ردگیری تانسورها (Source Code & Tensor Trace)}}: Class hierarchy, parameters, forward pass [B, N, D] -> [B, H, N, d], RoPE, GQA, LoRA adapters.
   \subsection*{{بخش ۴: تحلیل معماری سخت‌افزار و مدل سقف کارایی (Hardware & Roofline Analysis)}}: HBM vs SRAM (L1/L2), memory traffic, arithmetic intensity, memory-bound vs compute-bound regimes on NVIDIA RTX 4070.
   \subsection*{{بخش ۵: کارنامه تجربی و ارزیابی آزمایشگاهی (Empirical Results & Benchmark)}}: Table with genuine metrics from master_verified_results.json.
   \subsection*{{بخش ۶: تحلیل و تفسیر نتایج و رفتار رده‌های کلاسی (Interpretation & Error Analysis)}}: Scientific explanation of performance, F1 behavior, ablation effects.
   \subsection*{{بخش ۷: نقاط قوت، محدودیت‌ها و پیشنهادات توسعه (Strengths, Weaknesses & Future Scope)}}: Structured bullet points.
   (Preserve any existing tikzpicture diagrams from the original .tex file, if any).
4. Strict Rules for Persian LaTeX:
   - Latin parentheses: strictly \lr{{(...)}} (e.g. \lr{{(SVD)}}, \lr{{(GLUE)}}).
   - Escaped underscores: \_.
   - ZERO occurrences of legacy\_bert (use DistilBERT).
   - ZERO occurrences of A100 (use NVIDIA GeForce RTX 4070 Laptop (8GB GDDR6)).
   - ZERO occurrences of NeuralOps.
   - NO local \subsection*{{مراجع}} inside the file.
5. Use `write_to_file` with Overwrite=True to write the final tex content to the Target File.
6. When done, call `send_message` to report back to your parent."""
    })

print(json.dumps(subagents, indent=2))
