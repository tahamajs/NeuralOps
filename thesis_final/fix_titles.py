import os, glob

base_dir = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'
files = glob.glob(os.path.join(base_dir, '*.tex'))

replacements = {
    r'\subsection*{۱. فلسفه طراحی، پارادایم معماری و تمثیل ملموس}': r'\subsection*{ایده اصلی و مبانی تئوری}',
    r'\subsection*{۲. فرمول‌بندی و استنتاج عمیق ریاضی (Mathematical Derivation)}': r'\subsection*{مدل‌سازی ریاضی}',
    r'\subsection*{۳. کالبدشکافی معماری (Architectural Anatomy)}': r'\subsection*{تحلیل ساختار معماری}',
    r'\subsection*{۴. تحلیل پیچیدگی و محدودیت‌ها (Complexity \& Limitations)}': r'\subsection*{ارزیابی پیچیدگی محاسباتی و محدودیت‌ها}',
    r'\subsection*{۵. ارزیابی تجربی، بنچمارک و توجیه علمی نتایج}': r'\subsection*{بررسی نتایج آزمایشگاهی}',
    r'\subsection*{۶. دیاگرام معماری شکیل در لاتک (TikZ Architecture Flowchart)}': r'\subsection*{نمودار جریان داده‌ها و ساختار بلوکی}',
    r'\subsection*{۷. پیوست تصویری و پرامپت تولید هوش مصنوعی}': r'\subsection*{پیوست: شماتیک بصری}'
}

count = 0
for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1

print(f"Updated headers in {count} files.")
