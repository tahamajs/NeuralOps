import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings
import matplotlib.pyplot as plt

# تنظیم فونت فارسی (اختیاری، اگر روی مک مشکل داشتید حذف شود)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica Neue', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

def analyze_top_500_distribution(file_path):
    """بررسی توزیع برچسب‌ها در ۵۰۰ نمونه اول"""
    try:
        df = pd.read_csv(file_path)
        
        # پیدا کردن ستون‌ها
        text_cols = [c for c in df.columns if 'text' in c.lower()]
        label_cols = [c for c in df.columns if 'label' in c.lower()]
        
        if not text_cols or not label_cols:
            return None, "خطا: ستون text یا label یافت نشد"
        
        text_col = text_cols[0]
        label_col = label_cols[0]
        
        # انتخاب فقط ۵۰۰ نمونه اول
        df_sample = df.head(500).dropna(subset=[text_col, label_col])
        
        if len(df_sample) == 0:
            return None, "خطا: داده خالی پس از حذف سطرها با مقدار NaN"
            
        # شمارش برچسب‌ها
        label_counts = df_sample[label_col].value_counts().sort_index()
        total_samples = len(df_sample)
        
        # محاسبه درصد
        percentages = (label_counts / total_samples * 100).round(2)
        
        report = {
            "filename": Path(file_path).stem,
            "samples_analyzed": total_samples,
            "counts": label_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "is_balanced": True,
            "issues": []
        }
        
        # تحلیل تعادل (فقط برای دیتاست‌های دو کلاسه 0/1)
        if set(label_counts.index) <= {0, 1} and len(label_counts) == 2:
            count_0 = label_counts.get(0, 0)
            count_1 = label_counts.get(1, 0)
            
            ratio = min(count_0, count_1) / max(count_0, count_1)
            
            if ratio < 0.8: # اگر نسبت کمتر از 0.8 بود (مثلاً 70-30)
                report["is_balanced"] = False
                report["issues"].append(f"⚠️ عدم تعادل شدید: نسبت {ratio:.2f} (حدود {percentages.iloc[0]}% vs {percentages.iloc[1]}%)")
            else:
                report["issues"].append("✅ توزیع متعادل است.")
        else:
            report["issues"].append(f"ℹ️ چند کلاسه است ({len(label_counts)} کلاس).")

        return report, None
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def main():
    print("="*80)
    print("📊 DATASET DISTRIBUTION ANALYSIS (TOP 500 SAMPLES)")
    print("="*80)
    
    data_dir = "./data"
    csv_files = sorted(glob.glob(f"{data_dir}/*.csv"))
    
    if not csv_files:
        print(f"❌ هیچ فایل CSV در پوشه {data_dir} یافت نشد.")
        return

    reports = []
    
    # لیست برای رسم نمودار
    filenames = []
    class_0_pct = []
    class_1_pct = []
    colors = []
    
    print(f"📂 Found {len(csv_files)} files.\n")
    
    for file_path in csv_files:
        report, error = analyze_top_500_distribution(file_path)
        
        if error:
            print(f"❌ {Path(file_path).name}: {error}")
        elif report:
            reports.append(report)
            
            # نمایش گزارش متنی
            status_icon = "✅" if report["is_balanced"] else "⚠️"
            print(f"\n{status_icon} {report['filename']} ({report['samples_analyzed']} samples)")
            print(f"   Labels: {report['counts']}")
            print(f"   Percentages: {report['percentages']}")
            
            for issue in report['issues']:
                print(f"   → {issue}")
            
            # آماده‌سازی برای نمودار (فقط برای دیتاست‌های دو کلاسه)
            if set(report['counts'].keys()) <= {0, 1} and len(report['counts']) == 2:
                filenames.append(report['filename'])
                class_0_pct.append(report['percentages'].get(0, 0))
                class_1_pct.append(report['percentages'].get(1, 0))
                
                # رنگ سبز برای متعادل، قرمز برای نامتعادل
                colors.append('green' if report['is_balanced'] else 'red')
            else:
                filenames.append(report['filename'])
                class_0_pct.append(0)
                class_1_pct.append(0)
                colors.append('gray')

    # رسم نمودار
    if len([x for x in class_0_pct if x > 0]) > 0:
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(filenames))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, class_0_pct, width, label='Class 0', color=colors, alpha=0.7)
        bars2 = plt.bar(x + width/2, class_1_pct, width, label='Class 1', color=colors, alpha=0.7)
        
        plt.xlabel('Dataset Name (Top 500 Samples)')
        plt.ylabel('Percentage (%)')
        plt.title('Label Distribution Analysis (Top 500 Samples)\nGreen=Balanced, Red=Imbalanced, Gray=Multi-class')
        plt.xticks(x, filenames, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('distribution_report.png', dpi=300)
        print("\n🖼️ Graph saved as 'distribution_report.png'")
        plt.show()
    else:
        print("\n⚠️ No binary datasets found to generate a graph.")

if __name__ == "__main__":
    main()