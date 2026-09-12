import re

with open("master_thesis.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Replace NeuralOps occurrences
text = text.replace("NeuralOps: ", "")
text = text.replace("NeuralOps:", "")
text = text.replace("\\lr{NeuralOps}", "چارچوب پژوهش")
text = text.replace("NeuralOps", "چارچوب سیستم‌محور پژوهش")
text = text.replace("فریمورک منبع‌باز", "چارچوب سیستم‌محور")
text = text.replace("فریم‌ورک منبع‌باز", "چارچوب سیستم‌محور")
text = text.replace("فریم‌ورک منبع باز", "چارچوب سیستم‌محور")
text = text.replace("فریمورک منبع باز", "چارچوب سیستم‌محور")
text = text.replace("فریمورک", "چارچوب")
text = text.replace("فریم‌ورک", "چارچوب")

# Fix English Abstract
text = text.replace("This thesis presents \\textbf{چارچوب سیستم‌محور پژوهش}", "This thesis presents a comprehensive hardware-aware framework")
text = text.replace("This thesis presents \\textbf{NeuralOps}", "This thesis presents a comprehensive hardware-aware framework")
text = text.replace("چارچوب سیستم‌محور پژوهش Framework", "Hardware-Aware Framework")

# Clean any double "چارچوب چارچوب"
text = re.sub(r'چارچوب\s+چارچوب', 'چارچوب', text)

with open("master_thesis.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("NeuralOps removed completely from master_thesis.tex.")
