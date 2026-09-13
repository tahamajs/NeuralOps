import re

with open("master_thesis.tex", "r", encoding="utf-8") as f:
    content = f.read()

cm_block = """
\\section{تحلیل ماتریس درهم‌ریختگی \\lr{(Confusion Matrix)} برای ۵ ماژول برتر}
برای درک بهتر عملکرد تفکیکی مدل‌های برتر در طبقه‌بندی احساسات (دیتاست \\lr{SST-2})، ماتریس درهم‌ریختگی ۵ مدل برتر استخراج شده است:
\\begin{figure}[H]
    \\centering
    \\includegraphics[width=0.45\\textwidth]{img/confusion/Fusion_Attention_cm.png}
    \\includegraphics[width=0.45\\textwidth]{img/confusion/K_Means_Attention_cm.png}
    \\includegraphics[width=0.45\\textwidth]{img/confusion/Primal_Attention_cm.png}
    \\includegraphics[width=0.45\\textwidth]{img/confusion/SVD_LLM_V2_cm.png}
    \\includegraphics[width=0.45\\textwidth]{img/confusion/HGRN2_Gated_cm.png}
    \\caption{ماتریس درهم‌ریختگی ۵ مکانیزم برتر روی مجموعه داده ارزیابی}
    \\label{fig:top5_confusion_matrices}
\\end{figure}
تحلیل این ماتریس‌ها نشان می‌دهد که معماری قهرمان \\lr{(Fusion Attention)} کمترین میزان منفی کاذب \\lr{(False Negative)} را ثبت کرده است که دلیل اصلی برتری آن در معیار \\lr{Recall} می‌باشد.
"""

if "ماتریس درهم‌ریختگی" not in content:
    content = content.replace("\\chapter{نتیجه‌گیری و پیشنهادات}", cm_block + "\n\\chapter{نتیجه‌گیری و پیشنهادات}")
    with open("master_thesis.tex", "w", encoding="utf-8") as f:
        f.write(content)
        print("Injected Confusion Matrices.")
