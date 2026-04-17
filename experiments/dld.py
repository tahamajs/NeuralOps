import os
import time
from datasets import load_dataset, DatasetDict

# تنظیمات پیشرفته برای کاهش خطاهای شبکه
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # غیرفعال کردن انتقال سریع (گاهی باعث خطا می‌شود)
os.environ["HF_DATASETS_OFFLINE"] = "0"    # اطمینان از آنلاین بودن

print("🚀 شروع دانلود ۲۰ دیتاست NLP با مدیریت خطای شبکه...")

def safe_load_dataset(name, config=None, split="train", timeout=60):
    """تابع امن برای بارگذاری دیتاست با مدیریت خطا و تایم‌اوت"""
    try:
        print(f"Loading {name}...", end=" ")
        
        # تنظیم timeout برای درخواست‌های شبکه
        kwargs = {}
        if config:
            kwargs['split'] = split
            dataset = load_dataset(name, config=config, **kwargs)
        else:
            kwargs['split'] = split
            dataset = load_dataset(name, **kwargs)
            
        # بررسی اینکه آیا داده خالی نیست
        if isinstance(dataset, dict):
            size = sum(len(v) for v in dataset.values())
        else:
            size = len(dataset)
            
        print(f"✅ Success (Size: {size:,})")
        return True
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"❌ Failed: {error_msg}...")
        return False

# لیست ۲۰ دیتاست برتر NLP
datasets_to_load = [
    ("glue", "sst2"),       # Sentiment Analysis
    ("glue", "mrpc"),       # Paraphrase Detection
    ("glue", "qnli"),       # Question NLI
    ("glue", "stsb"),       # Semantic Textual Similarity
    ("glue", "mnli"),       # Multi-Genre NLI
    
    # دیتاست‌های حجیم که ممکن است نیاز به وقت بیشتری داشته باشند
    ("squad_v2", None),     # Question Answering
    ("wikitext", "wikitext-2-raw-v1"), 
    ("c4", "en"),           # Colossal Clean Crawled Corpus (بسیار بزرگ)
    ("dolly", "databricks-dolly-15k"), 
    ("alpaca", "tatsu-lab/alpaca"),    
    ("openwebtext", None),  
    ("yelp_review_full", None), 
    ("imdb", None),         
    ("trec", "trec_50"),    
    ("ag_news", None),      
    ("hotpotqa", "fullwiki"), 
    ("boolq", None),        
    ("rte", None),          
    ("wnli", None),         
    ("qqp", None),          
]

success_count = 0
total_count = len(datasets_to_load)

for i, ds_config in enumerate(datasets_to_load):
    dataset_name = ds_config[0]
    config_name = ds_config[1] if len(ds_config) > 1 else None
    
    # تلاش مجدد تا 3 بار با فاصله زمانی
    success = False
    attempts = 0
    while not success and attempts < 3:
        success = safe_load_dataset(dataset_name, config_name, split="train")
        if not success:
            attempts += 1
            if attempts < 3:
                print(f"  ⏳ Retry {attempts}/3 in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  ❌ Permanently failed after 3 attempts.")
    
    if success:
        success_count += 1

print(f"\n⚡ پایان مرحله اول: {success_count}/{total_count} دیتاست با موفقیت بارگذاری شدند.")

# --- بخش دوم: بنچمارک‌های LLM ---
print("\n🔥 شروع دانلود ۲۰ بنچمارک LLM...")

benchmarks_to_load = [
    ("mmlu", None),           
    ("hellaswag", None),      
    ("arc", "challenge"),     
    ("gsm8k", "main"),        
    ("truthfulqa", "multiple_choice"), 
    ("cot_collection", None), 
    ("math", None),           
    ("human_eval", None),     
    ("mbpp", None),           
    ("big_bench_hard", "logic_game"), 
    ("pile", "pile_pile_10k"), # نسخه کوچک پیل
    ("lambada", "standard"),  
    ("piqa", None),           
    ("winogrande", "winogrande_xl"), 
    ("sciq", None),           
    ("logiqa", None),         
    ("races", None),          
    ("quac", None),           
    ("drop", None),           
    ("copa", None),           
]

bench_success = 0
bench_total = len(benchmarks_to_load)

for i, bench_config in enumerate(benchmarks_to_load):
    bench_name = bench_config[0]
    bench_config_name = bench_config[1] if len(bench_config) > 1 else None
    
    # تلاش مجدد
    success = False
    attempts = 0
    while not success and attempts < 3:
        try:
            print(f"[{i+1}/{bench_total}] Loading {bench_name}...", end=" ")
            if bench_config_name:
                # برای بنچمارک‌ها معمولاً از split validation یا test استفاده می‌شود
                dataset = load_dataset(bench_name, bench_config_name, split="validation")
            else:
                dataset = load_dataset(bench_name, split="test")
            
            size = len(dataset)
            print(f"✅ Success (Size: {size:,})")
            success = True
            bench_success += 1
        except Exception as e:
            error_msg = str(e)[:60]
            print(f"❌ Failed: {error_msg}...")
            attempts += 1
            if attempts < 3:
                time.sleep(5)
            else:
                print(f"  ⚠️ Skipped permanently.")

print(f"\n✅ تمام فرآیند تکمیل شد. مجموع موفقیت‌ها: {success_count + bench_success}")
