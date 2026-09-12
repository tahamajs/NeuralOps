import os
import glob
import re

tex_files = glob.glob("attentions_latex/individual_modules/*.tex")

print(f"Found {len(tex_files)} module files to clean and format.")

count = 0

for file_path in tex_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    mod_name = os.path.basename(file_path).replace(".tex", "")
    escaped_mod = mod_name.replace("_", "\\_")

    # 1. Fix flipped parentheses: (\lr{Word}) -> \lr{(Word)}
    content = re.sub(r'\(\\lr\{([^}]+)\}\)', r'\\lr{(\1)}', content)
    
    # 2. Fix itemized colons for key metrics
    content = re.sub(r'\\item\s*\\textbf\{([^}]+)\s*\\lr\{(\([^)]+\))\}:\s*\}', r'\\item \\textbf{\1} \\lr{\2}\\rl{:} ', content)
    content = re.sub(r'\\item\s*\\textbf\{([^}]+)\}\s*\\lr\{(\([^)]+\))\}:\s*', r'\\item \\textbf{\1} \\lr{\2}\\rl{:} ', content)
    content = re.sub(r'\\item\s*\\textbf\{([^}]+)\s*\(\\lr\{([^}]+)\}\):\s*\}', r'\\item \\textbf{\1} \\lr{(\2)}\\rl{:} ', content)
    content = re.sub(r'\\item\s*\\textbf\{([^}]+)\}\s*\(\\lr\{([^}]+)\}\):\s*', r'\\item \\textbf{\1} \\lr{(\2)}\\rl{:} ', content)

    # Specific common ones
    content = content.replace("اوج مصرف حافظه گرافیکی (Peak VRAM):", "اوج مصرف حافظه گرافیکی \\lr{(Peak VRAM)}\\rl{:}")
    content = content.replace("نرخ توان عملیاتی (Throughput):", "نرخ توان عملیاتی \\lr{(Throughput)}\\rl{:}")
    content = content.replace("شاخص ارزیابی آماری (F1 Score):", "شاخص ارزیابی آماری \\lr{(F1 Score)}\\rl{:}")
    content = content.replace("مهار عطش پهنای باند حافظه (Memory-Bound Relief):", "مهار عطش پهنای باند حافظه \\lr{(Memory-Bound Relief)}\\rl{:}")
    content = content.replace("صرفه‌جویی بنیادین در بافر توکن‌ها (KV-Cache):", "صرفه‌جویی بنیادین در بافر توکن‌ها \\lr{(KV-Cache)}\\rl{:}")

    # 3. Fix numbers with commas and percentages in text
    # e.g. 35,800 or 15,400 -> \lr{35,800}
    content = re.sub(r'(?<!\\lr\{)(\b\d{1,3},\d{3}\b)', r'\\lr{\1}', content)
    # e.g. 91.86 درصد -> \lr{91.86\%}
    content = re.sub(r'(?<!\\lr\{)(\b\d{1,3}\.\d{1,3}\b)\s*درصد', r'\\lr{\1\\%}', content)
    # e.g. 430.0 مگابایت -> \lr{430.0 MB}
    content = re.sub(r'(?<!\\lr\{)(\b\d{1,4}(?:\.\d{1,2})?\b)\s*مگابایت', r'\\lr{\1 MB}', content)
    # e.g. tok/s or توکن بر ثانیه
    content = re.sub(r'(?<!\\lr\{)(\b\d{1,3}(?:,\d{3})*\b)\s*توکن بر ثانیه', r'\\lr{\1 tok/s}', content)

    # 4. Fix GPU and Hardware parentheses
    content = re.sub(r'\bGPU\b', r'\\lr{GPU}', content)
    content = re.sub(r'\(مانند\s*\\lr\{NVIDIA[^}]+\}\):', r'(نظیر کارت گرافیک \\lr{NVIDIA RTX 4070}):', content)
    content = re.sub(r'\(مانند\s*NVIDIA[^)]+\):', r'(نظیر کارت گرافیک \\lr{NVIDIA RTX 4070}):', content)

    # 5. Fix framebox brackets flipping
    content = re.sub(
        r'\\textbf\{\[کادر الحاق تصویر گرافیکی سه‌بعدی ماژول[^\\]*\\lr\{[^}]+\}\]\}',
        f'\\\\textbf{{طرح‌واره معماری و جریان تانسوری ماژول \\\\lr{{{escaped_mod}}}}}\\\\[0.3cm]\\\\small \\\\lr{{(Tensor Flow & Architectural Schematic for {escaped_mod})}}',
        content
    )
    content = re.sub(
        r'\\textbf\{\[کادر الحاق تصویر گرافیکی سه‌بعدی ماژول[^\]]+\]\}',
        f'\\\\textbf{{طرح‌واره معماری و جریان تانسوری ماژول \\\\lr{{{escaped_mod}}}}}\\\\[0.3cm]\\\\small \\\\lr{{(Tensor Flow & Architectural Schematic for {escaped_mod})}}',
        content
    )

    # 6. Ensure captions use escaped_mod inside \lr
    content = re.sub(
        r'\\caption\{شماتیک و جریان داده تانسوری در ماژول\s+([^}]+)\}',
        f'\\\\caption{{شماتیک و جریان داده تانسوری در ماژول \\\\lr{{{escaped_mod}}}}}',
        content
    )

    if content != original:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1

print(f"Successfully beautified and fixed {count} module files.")
