import glob
import re

files = glob.glob("attentions_latex/individual_modules/*.tex")
count = 0

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        content = fp.read()

    original = content
    # 1. Fix unclosed subsubsection brace
    content = content.replace(
        "\\subsubsection*{۵. ارزیابی تجربی، بنچمارک و توجیه علمی نتایج\n\\begin{table}",
        "\\subsubsection*{۵. ارزیابی تجربی، بنچمارک و توجیه علمی نتایج}\n\\begin{table}"
    )

    # 2. Fix nested \lr{\lr{...}}
    content = re.sub(r'\\lr\{\\lr\{([^}]+)\}\}', r'\\lr{\1}', content)
    content = re.sub(r'\\lr\{\(\\lr\{([^}]+)\}\)\}', r'\\lr{(\1)}', content)

    # 3. Clean table header in all tables
    content = content.replace(
        r"\textbf{مدل پایه \lr{(\lr{Base Model})}} & \textbf{دقت \lr{(\lr{Accuracy})}} & \textbf{شاخص \lr{F1}} & \textbf{حافظه \lr{(\lr{Peak VRAM})}} & \textbf{توان \lr{(\lr{Throughput})}} & \textbf{وضعیت}",
        r"\textbf{مدل پایه} \lr{(Base Model)} & \textbf{دقت} \lr{(Accuracy)} & \textbf{شاخص \lr{F1}} & \textbf{حافظه} \lr{(Peak VRAM)} & \textbf{توان} \lr{(Throughput)} & \textbf{وضعیت}"
    )
    content = content.replace(
        r"\textbf{مدل پایه (\lr{Base Model})} & \textbf{دقت (\lr{Accuracy})} & \textbf{شاخص \lr{F1}} & \textbf{حافظه (\lr{Peak VRAM})} & \textbf{توان (\lr{Throughput})} & \textbf{وضعیت}",
        r"\textbf{مدل پایه} \lr{(Base Model)} & \textbf{دقت} \lr{(Accuracy)} & \textbf{شاخص \lr{F1}} & \textbf{حافظه} \lr{(Peak VRAM)} & \textbf{توان} \lr{(Throughput)} & \textbf{وضعیت}"
    )

    if content != original:
        with open(f, "w", encoding="utf-8") as fp:
            fp.write(content)
        count += 1

print(f"Cleaned table syntax in {count} files.")
