with open('thesis_final/master_thesis.tex', 'r', encoding='utf-8') as f:
    content = f.read()

old_nodes = r"""\node[phase] (p1) {\textbf{گام اول: تزریق درجا (Drop-in Injection)}\\جایگزینی لایه توجه بدون تغییر معماری کلی};
\node[phase, below=of p1] (p2) {\textbf{گام دوم: نگاشت تانسورها (Weight Alignment)}\\استخراج ماتریس‌ها، فاکتورسازی SVD و لایه LoRA};
\node[phase, below=of p2] (p3) {\textbf{گام سوم: بازیابی سریع (Recovery Phase)}\\کالیبراسیون ۵۰ تا ۲۵۰ گام با نرخ یادگیری $1e-4$};
\node[phase, below=of p3] (p4) {\textbf{گام چهارم: پروفایلینگ سخت‌افزاری (Hardware Profiling)}\\اندازه‌گیری VRAM، Throughput و ارزیابی Quality Gate};"""

new_nodes = r"""\node[phase] (p1) {\textbf{گام اول: تزریق درجای ماژول \lr{(Drop-in Injection)}}\\جایگزینی مستقیم لایه توجه بدون ایجاد تغییر در معماری پایه};
\node[phase, below=of p1] (p2) {\textbf{گام دوم: نگاشت پارامتری تانسورها \lr{(Weight Alignment)}}\\تجزیه مقادیر منفرد \lr{(SVD)} و مقداردهی ماتریس‌های کم‌مرتبه \lr{(LoRA)}};
\node[phase, below=of p2] (p3) {\textbf{گام سوم: فاز بازیابی سریع \lr{(Recovery Phase)}}\\کالیبراسیون شبکه در ۵۰ الی ۲۵۰ گام با نرخ یادگیری $\lr{1e-4}$};
\node[phase, below=of p3] (p4) {\textbf{گام چهارم: پروفایلینگ سخت‌افزاری \lr{(Hardware Profiling)}}\\سنجش اوج مصرف حافظه، توان عملیاتی و ارزیابی دروازه کیفیت \lr{(Quality Gate)}};"""

if old_nodes in content:
    content = content.replace(old_nodes, new_nodes)
    with open('thesis_final/master_thesis.tex', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Nodes replaced successfully.")
else:
    print("Old nodes not found.")
