# گزارش جامع پاکسازی و اعتبارسنجی سیستم (CLEANUP REPORT)
**تاریخ تدوین:** ۲۴ سپتامبر ۲۰۲۶  
**محیط اجرا:** macOS Darwin / مخزن تحقیقاتی پروژه ارزیابی مکانیزم‌های توجه و فشرده‌سازی SVD

---

## ۱. نتایج حذف ماژول‌های جعلی (فقدان کد سورس پایتون)
تمامی ۲۳ فایل لاتک زیر که فاقد هرگونه پیاده‌سازی متناظر در پوشه‌های `Codes/framework/` بودند، از مسیر `thesis_final/attentions_latex/individual_modules/` حذف گردیدند:

- `async_spec_attention.tex`
- `block_sparse_attention.tex`
- `dynamic_entropy_attention.tex`
- `fourier_attention.tex`
- `generalized_attention.tex`
- `hyper_attention.tex`
- `memory_efficient_attention.tex`
- `poly_attention.tex`
- `prefix_attention.tex`
- `prisma_attention.tex`
- `reformer_attention.tex`
- `retention.tex`
- `ring_attention.tex`
- `sinkhorn_attention.tex`
- `structured_attention.tex`
- `swin_attention.tex`
- `syntactic_attention.tex`
- `tensor_product_attention.tex`
- `tree_attention.tex`
- `truncated_attention.tex`
- `uniform_attention.tex`
- `variable_rate_attention.tex`
- `wave_attention.tex`

**آمار حذف:**
- **تعداد کل فایل‌های حذف شده:** ۲۳ فایل
- **تعداد فایل‌های باقیمانده در individual_modules:** ۱۴۲ فایل

---

## ۲. پاکسازی و اصلاح رانرهای آزمایشگاهی (`Codes/runners/`)
تمامی اسکریپت‌های رانر `run_all_4_*.py` و اسکریپت‌های کمکی ارزیابی که حاوی مسیرهای مطلق محیط کاربری ویندوزی (`D:\svd_project` و مفسر آناکوندا `C:\Users\Hadi Schumacher\...`) بودند، با متغیرهای استاندارد و قابل حمل جایگزین شدند:

- **تعداد فایل‌های رانر اصلاح‌شده:** **۲۳۱ فایل**
- **بررسی صحت پس از اصلاح:**
  ```bash
  grep -l "D:\\\\svd_project" Codes/runners/*.py | wc -l
  # خروجی: 0
  grep -l "Hadi Schumacher" Codes/runners/*.py | wc -l
  # خروجی: 0
  ```
- **وضعیت فعلی آلودگی مسیرها:** **۰ (کاملاً پاکسازی شده و قابل حمل)**.

---

## ۳. کامپایل مجدد و اعتبارسنجی سند پایان‌نامه (`master_thesis_lite.pdf`)
پس از حذف فایل‌های جعلی و پایدارسازی مونوگراف‌ها، پایان‌نامه طی دو مرحله متوالی با `xelatex` کامپایل شد:

- **تعداد صفحات واقعی PDF:** **۴۵۸ صفحه**
- **حجم دقیق فایل:** ۱,۶۳۱,۷۹۲ بایت (~ ۱.۵۵ مگابایت)
- **وضعیت کامپایل:** موفق (Exit Code 0)، بدون خطای حافظه، بدون خطای شناورها، بدون تداخل بسته زی‌پرشین.

---

## ۴. جمع‌بندی واقعیت‌های پایگاه داده و ماژول‌ها
1. **تعداد ماژول‌های فعال و مدون در فصل چهارم رساله:** **۸۴ ماژول یکتا و معتبر**.
2. **تعداد مقالات مستقل کامپایل‌شده در `paper/PerModulePaper`:** **۲۱۷ مقاله PDF**.
3. **پایگاه داده مرجع نتایج:** `thesis_final/master_verified_results.json` شامل ۳۳۱ رکورد ارزیابی استخراج‌شده از آزمایش‌ها.
