import os, glob

base_dir = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'
files = glob.glob(os.path.join(base_dir, '*.tex'))

count = 0
for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # We want to replace:
    # \begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=نتایج ارزیابی تجربی و تحلیل ریشه‌ای ماژول \lr{...}]
    # with mdframed.
    
    import re
    # Match the begin tag and capture the title
    pattern = r'\\begin\{tcolorbox\}\[colback=blue!5!white,colframe=blue!75!black,title=(.*?)\]'
    
    def replacer(match):
        title = match.group(1)
        return (f"\\begin{{mdframed}}[roundcorner=4pt, backgroundcolor=blue!5!white, "
                f"linecolor=blue!75!black, linewidth=1.2pt, "
                f"frametitle={{{title}}}, frametitlerule=true, "
                f"frametitlebackgroundcolor=blue!10!white, frametitlefont=\\bfseries]")
    
    content = re.sub(pattern, replacer, content)
    
    content = content.replace(r'\end{tcolorbox}', r'\end{mdframed}')
        
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1

print(f"Updated tcolorbox to mdframed in {count} files.")
