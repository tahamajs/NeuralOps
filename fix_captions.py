import os
import glob
import re

directory = 'thesis_final/attentions_latex/individual_modules'
files = glob.glob(os.path.join(directory, '*.tex'))

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to replace the captions to move the \lr{} to the end of the caption or modify it to avoid breaking
    # Current captions:
    # 1. \caption{دیاگرام ساختار معماری و جریان تانسوری ماژول \lr{module}}
    # 2. \caption{جزئیات تکمیلی پردازش و نگاشت تانسورها در ماژول \lr{module}}
    # 3. \caption{نمودار ارزیابی مقایسه‌ای سه‌شاخصی دقت، اوج مصرف حافظه و توان عملیاتی مکانیزم \lr{module} در برابر خط مبنا}
    
    # Actually, a common trick in XePersian to prevent caption break is to wrap the English text in \rl{\lr{module}} or just avoiding it in the middle of text.
    # Let's change caption 3 to:
    # \caption{نمودار ارزیابی مقایسه‌ای سه‌شاخصی مکانیزم در برابر خط مبنا (\lr{module})}
    
    # Or just replace the exact problematic caption 3:
    content = re.sub(
        r'\\caption\{نمودار ارزیابی مقایسه‌ای سه‌شاخصی دقت، اوج مصرف حافظه و توان عملیاتی مکانیزم \\lr\{([^}]+)\} در برابر خط مبنا\}',
        r'\\caption{نمودار ارزیابی مقایسه‌ای سه‌شاخصی مکانیزم در برابر خط مبنا (\\lr{\1})}',
        content
    )
    
    content = re.sub(
        r'\\caption\{دیاگرام ساختار معماری و جریان تانسوری ماژول \\lr\{([^}]+)\}\}',
        r'\\caption{دیاگرام ساختار معماری و جریان تانسوری ماژول (\\lr{\1})}',
        content
    )
    
    content = re.sub(
        r'\\caption\{جزئیات تکمیلی پردازش و نگاشت تانسورها در ماژول \\lr\{([^}]+)\}\}',
        r'\\caption{جزئیات تکمیلی پردازش و نگاشت تانسورها در ماژول (\\lr{\1})}',
        content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Updated all captions.")
