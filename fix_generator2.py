with open('generate_all_12_sections.py') as f:
    c = f.read()

# Fix extra closing brace in caption
c = c.replace(r'مقایسه عملکرد ماژول \lr{" + mod_name_escaped + r"} با خط پایه بر روی مدل " + tested_model_short + r"}}"',
              r'مقایسه عملکرد ماژول \lr{" + mod_name_escaped + r"} با خط پایه بر روی مدل " + tested_model_short + r"}"')

# Escape all ampersands in FAMILY_DATA
c = c.replace("'SVD & Low-Rank Spectral Decomposition'", r"'SVD \& Low-Rank Spectral Decomposition'")
c = c.replace("'State-Space Models & Linear Recurrence'", r"'State-Space Models \& Linear Recurrence'")
c = c.replace("'Dynamic & Block Sparse Routing Attention'", r"'Dynamic \& Block Sparse Routing Attention'")
c = c.replace("'Hardware-Fused & Quantized Attention (FP8/Triton)'", r"'Hardware-Fused \& Quantized Attention (FP8/Triton)'")
c = c.replace("'Mixture of Experts & Ensemble Fusion'", r"'Mixture of Experts \& Ensemble Fusion'")
c = c.replace("Roofline Model & IO Complexity", r"Roofline Model \& IO Complexity")

# Escape & in references
c = c.replace("& Young, G.", r"\& Young, G.")
c = c.replace("& Tropp, J. A.", r"\& Tropp, J. A.")
c = c.replace("& Dao, T.", r"\& Dao, T.")
c = c.replace("& Ré, C.", r"\& Ré, C.")
c = c.replace("& Levskaya, A.", r"\& Levskaya, A.")
c = c.replace("& Cohan, A.", r"\& Cohan, A.")
c = c.replace("& Patterson, D.", r"\& Patterson, D.")
c = c.replace("& Shazeer, N.", r"\& Shazeer, N.")

# Also in Architecture & Implementation section title:
c = c.replace(r"\subsection*{بخش ۵: معماری و پیاده‌سازی (Architecture & Implementation)}",
              r"\subsection*{بخش ۵: معماری و پیاده‌سازی (Architecture \& Implementation)}")
c = c.replace(r"\subsection*{بخش ۹: نقاط قوت و ضعف (Strengths & Weaknesses)}",
              r"\subsection*{بخش ۹: نقاط قوت و ضعف (Strengths \& Weaknesses)}")
c = c.replace(r"\subsection*{بخش ۱۱: تصاویر و نمودارها (Figures & Charts)}",
              r"\subsection*{بخش ۱۱: تصاویر و نمودارها (Figures \& Charts)}")

with open('generate_all_12_sections.py', 'w') as f:
    f.write(c)

print("Updated generate_all_12_sections.py with all fixes.")
