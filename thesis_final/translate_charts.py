import re

with open('generate_charts_v2_en.py', 'r', encoding='utf-8') as f:
    code = f.read()

translations = {
    "'میانگین دقت (%)'": "'Average Accuracy (%)'",
    "'۳۰ ماژول برتر بر اساس میانگین دقت در ۳ مدل'": "'Top 30 Modules by Average Accuracy across 3 Models'",
    "'خط مبنا'": "'Baseline'",
    "'وضعیت'": "'Status'",
    "'تعداد آزمایش‌ها'": "'Number of Trials'",
    "'توزیع نتایج PASS/FAIL بر اساس خانواده معماری'": "'PASS/FAIL Distribution by Architecture Family'",
    "'دقت (%)'": "'Accuracy (%)'",
    "'نقشه حرارتی دقت: ۴۰ ماژول برتر × ۳ مدل'": "'Accuracy Heatmap: Top 40 Modules × 3 Models'",
    "'آستانه ۸۰%'": "'80% Threshold'",
    "'تعداد مدل‌های آزمایش‌شده'": "'Number of Models Tested'",
    "'رابطه تعداد مدل‌های آزمایش‌شده و میانگین دقت'": "'Average Accuracy vs. Number of Models Tested'",
    "'توزیع دقت در هر خانواده معماری'": "'Accuracy Distribution per Architecture Family'",
    "'مقایسه دقت ۲۰ ماژول برتر به تفکیک مدل زبانی'": "'Top 20 Modules Accuracy Comparison by Base Model'",
    "'نرخ PASS'": "'PASS Rate'",
    "'تعداد ماژول'": "'Module Count'",
    "'مقایسه رادار خانواده‌های معماری'": "'Radar Comparison of Architecture Families'",
    "f'میانگین: {np.mean(all_accs):.1f}%'": "f'Mean: {np.mean(all_accs):.1f}%'",
    "'توزیع دقت در ۱۵۰ ماژول بررسی‌شده'": "'Accuracy Distribution Across 150 Evaluated Modules'",
    "f'{c} ماژول\\n≥{t}%'": "f'{c} Modules\\n≥{t}%'",
    "'آستانه دقت (%)'": "'Accuracy Threshold (%)'",
    "'تعداد ماژول‌های برتر از آستانه'": "'Number of Modules Above Threshold'",
    "'تعداد ماژول‌ها به ازای هر آستانه دقت'": "'Cumulative Number of Modules by Accuracy Threshold'",
    "'میانگین دقت به تفکیک مدل زبانی پایه'": "'Mean Accuracy by Base Language Model'",
    "f'{k}\\n({v} ماژول)'": "f'{k}\\n({v} Modules)'",
    "'توزیع ماژول‌ها بر اساس خانواده معماری'": "'Distribution of Modules by Architecture Family'",
    "'۱۵ ماژول برتر'": "'Top 15 Modules'",
    "'۱۵ ماژول ضعیف‌تر'": "'Bottom 15 Modules'",
    "'مقایسه بهترین و ضعیف‌ترین ماژول‌ها'": "'Comparison of Best and Worst Modules'",
    "'خط برابری y=x'": "'Line of Equality y=x'",
    "'دقت روی DistilBERT (%)'": "'Accuracy on DistilBERT (%)'",
    "'دقت روی RoBERTa (%)'": "'Accuracy on RoBERTa (%)'",
    "'مقایسه دقت DistilBERT در برابر RoBERTa'": "'Accuracy Comparison: DistilBERT vs RoBERTa'",
    "'خط برابری'": "'Line of Equality'",
    "'دقت روی GPT-2 (%)'": "'Accuracy on GPT-2 (%)'",
    "'مقایسه دقت GPT-2 در برابر RoBERTa'": "'Accuracy Comparison: GPT-2 vs RoBERTa'",
    "'۵۰٪ آستانه پایه'": "'50% Baseline Threshold'",
    "'نرخ موفقیت (PASS) به تفکیک خانواده معماری'": "'Success Rate (PASS) by Architecture Family'",
    "'توزیع دقت به تفکیک مدل زبانی (Violin Plot)'": "'Accuracy Distribution by Base Model (Violin Plot)'",
    "'الف) توزیع دقت کلی'": "'A) Overall Accuracy Distribution'",
    "'تعداد'": "'Count'",
    "f'μ={np.mean(all_accs):.1f}%'": "f'μ={np.mean(all_accs):.1f}%'",
    "'ب) توزیع کلی وضعیت آزمایش‌ها'": "'B) Overall Test Status Distribution'",
    "'ج) میانگین دقت به تفکیک مدل'": "'C) Mean Accuracy by Model'",
    "'د) ۱۰ ماژول برتر'": "'D) Top 10 Modules'",
    "'داشبورد جامع نتایج ارزیابی ۱۵۰ ماژول'": "'Comprehensive Evaluation Dashboard of 150 Modules'",
    "'میانگین دقت به تفکیک خانواده معماری'": "'Mean Accuracy by Architecture Family'",
}

for fa, en in translations.items():
    code = code.replace(fa, en)

with open('generate_charts_v2_en.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Translations applied.")
