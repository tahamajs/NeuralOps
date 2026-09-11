with open('master_thesis.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update section title and description
old_sec = r"\section{تحلیل نموداری جامع: تمامی نتایج بنچ‌مارک ۱۵۰ ماژول}"
new_sec = r"\section{تحلیل نموداری جامع ماژول‌های برتر و تاییدشده \lr{(Validated PASS Modules)}}"
text = text.replace(old_sec, new_sec)

old_desc = r"در این بخش، کلیه نتایج استخراج‌شده از خط لوله بنچ‌مارک یکپارچه روی بیش از ۱۵۰ مکانیزم توجه و فشرده‌سازی به صورت نمودارهای تحلیلی چندگانه ارائه می‌گردند. تمامی ارقام مستقیماً از فایل‌های نتایج تجربی \lr{(master\_aggregated\_results.json)} استخراج شده‌اند. خط مبنا در تمام نمودارها نشان‌دهنده توجه کامل استاندارد با دقت ۱.۹۱٪ است."
new_desc = r"در این بخش، تمامی ماژول‌های ناموفق، معیوب، دارای خطای محاسباتی یا دچار فروپاشی بازنمایی معنایی (وضعیت‌های \lr{FAIL} و \lr{PASS\_LOW}) به کلی از نتایج و تحلیل‌ها فیلتر و حذف گردیده‌اند. نمودارهای تحلیلی زیر منحصراً بر پایه ماژول‌های پیروزمند و تاییدشده با وضعیت قطعی \lr{PASS} و دقت بالا استخراج و رسم شده‌اند. همچنین جهت خوانایی استاندارد و انطباق کامل با معیارهای بین‌المللی، کلیه نمودارها به زبان انگلیسی رسمی تولید شده‌اند."
text = text.replace(old_desc, new_desc)

# 2. Update captions for charts that changed focus
text = text.replace(r"\caption{توزیع وضعیت‌های \lr{PASS}، \lr{PASS\_LOW} و \lr{FAIL} بر اساس هر خانواده معماری}",
                    r"\caption{توزیع ماژول‌های برتر تاییدشده \lr{(PASS)} به تفکیک خانواده معماری}")

text = text.replace(r"\caption{نرخ موفقیت \lr{(PASS Rate)} به تفکیک خانواده معماری}",
                    r"\caption{نسبت ماژول‌های با دقت بالای ۸۰٪ در هر خانواده معماری}")

text = text.replace(r"\caption{مقایسه ۱۵ ماژول برتر در برابر ۱۵ ماژول ضعیف‌تر از منظر میانگین دقت}",
                    r"\caption{دقت ۱۵ ماژول برتر ارزیابی‌شده در مقایسه با خط مبنای استاندارد}")

with open('master_thesis.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated master_thesis.tex captions and description successfully!")
