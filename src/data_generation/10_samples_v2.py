"""
🎯 Advanced Multi-Task Reasoning Dataset Generator
===================================================
10 Distinct Reasoning Types with Unique Cognitive Paths

Each task requires a DIFFERENT reasoning pathway:
1. DEDUCTION        → Rule + Case → Conclusion (If A→B, A=True → B=True)
2. INDUCTION        → Cases → General Rule (All observed A are B → All A are B)
3. ABDUCTION       → Observation → Best Explanation (Phenomenon → Most likely cause)
4. ANALOGY          → A:B :: C:?  (Find relation between C and something like A:B)
5. COUNTERFACTUAL  → What if... (Contrary to facts → Hypothetical outcome)
6. CAUSAL           → Cause → Effect (X leads to Y because...)
7. TEMPORAL         → Time relations (Before/After/Sequence)
8. SPATIAL          → Location relations (North/South/Inside/Behind...)
9. QUANTITATIVE     → Mathematical operations (Calculate/Compare/Measure)
10. SOCIAL          → Intent/Belief/Desire (Understand people's minds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
import string
import re
from typing import List, Dict, Tuple, Any
import json

random.seed(42)
np.random.seed(42)

# =============================================================================
# KNOWLEDGE BASES
# =============================================================================

PERSONS = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
ANIMALS = ["dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant"]
OBJECTS = ["book", "car", "phone", "computer", "chair", "table", "lamp", "box"]
LOCATIONS = ["kitchen", "bedroom", "garden", "office", "store", "school", "park", "library"]
CONTAINERS = ["box", "bag", "room", "house", "car", "cupboard", "drawer", "cabinet"]

# =============================================================================
# TASK 1: DEDUCTIVE REASONING (قیاس منطقی)
# =============================================================================
def generate_deductive_dataset(n: int) -> pd.DataFrame:
    """
    🧠 DEDUCTIVE REASONING
    ======================
    مسیر: premise (قاعده کلی) + fact (مورد خاص) → conclusion (نتیجه)
    
    مثال:
    - قاعده: "همه پرندگان پرواز می‌کنند"
    - مورد: "سیمرغ یک پرنده است"
    - نتیجه: "سیمرغ پرواز می‌کند" ✓
    
    مسیر استدلالی: Rule → Application → Conclusion
    """
    data = []
    
    # قوانین کلی (Universal Rules)
    rules = [
        ("همه {animal}ها {verb} می‌کنند", "یک {animal} است", "این {animal} {verb} می‌کند"),
        ("هیچ {animal} نمی‌تواند {verb}", "این یک {animal} است", "این {animal} نمی‌تواند {verb}"),
        ("هر {animal} یک {property} است", "فلان {animal} یک {animal} است", "فلان {animal} یک {property} است"),
        ("اگر {condition}، آنگاه {consequence}", "{condition} است", "پس {consequence}"),
        ("فقط {animal}ها می‌توانند {verb}", "این موجود یک {animal} است", "این موجود می‌تواند {verb}"),
    ]
    
    facts_positive = [
        ("پرندگان", "پرواز می‌کنند", "یک کبوتر", "کبوتر", "پرواز می‌کند"),
        ("ماهی‌ها", "شنا می‌کنند", "یک ماهی", "ماهی", "شنا می‌کند"),
        ("گربه‌ها", "پنجه می‌کشند", "یک گربه", "گربه", "پنجه می‌کشند"),
        ("انسان‌ها", "فکر می‌کنند", "یک انسان", "انسان", "فکر می‌کند"),
        ("خودروها", "حرکت می‌کنند", "یک ماشین", "ماشین", "حرکت می‌کند"),
    ]
    
    facts_negative = [
        ("پرندگان", "شنا می‌کنند", "یک کبوتر", "کبوتر", "شنا نمی‌کند"),
        ("ماهی‌ها", "پرواز می‌کنند", "یک ماهی", "ماهی", "پرواز نمی‌کند"),
        ("سنگ‌ها", "حرکت می‌کنند", "یک سنگ", "سنگ", "حرکت نمی‌کند"),
    ]
    
    conditional_rules = [
        ("باران ببارد", "زمین خیس می‌شود", "باران می‌بارد", "زمین خیس می‌شود"),
        ("برق قطع شود", "چراغ‌ها خاموش می‌شوند", "برق قطع شده", "چراغ‌ها خاموش شده‌اند"),
        ("دما زیر صفر شود", "آب یخ می‌زند", "دما زیر صفر است", "آب یخ زده"),
        ("کسی مرتکب جرم شود", "باید مجازات شود", "او مرتکب جرم شده", "او باید مجازات شود"),
        ("امتحان ندهی", "مردود می‌شوی", "او امتحان نداده", "او مردود شده"),
    ]
    
    # Positive deductions (entailment)
    for _ in range(n // 3):
        # Type 1: All A are B + This is A → This is B
        animal, verb, fact, name, conclusion = random.choice(facts_positive)
        rule = f"قاعده: همه {animal}ها {verb}"
        fact_text = f"واقعیت: {fact}"
        question = f"نتیجه چیست؟"
        text = f"{rule}. {fact_text}. {question}"
        data.append({"text": text, "label": 1, "task_type": "deductive_reasoning"})
        
        # Type 2: If A then B + A → B
        cond, cons, fact_text, conclusion = random.choice(conditional_rules)
        rule = f"قاعده: اگر {cond}، آنگاه {cons}"
        fact = f"واقعیت: {fact_text}"
        question = f"چه نتیجه‌ای می‌توان گرفت؟"
        text = f"{rule}. {fact}. {question}"
        data.append({"text": text, "label": 1, "task_type": "deductive_reasoning"})
    
    # Negative deductions (no entailment)
    for _ in range(n // 3):
        # Type 3: All A are B + This is not A → Unknown
        animal, verb, fact, name, conclusion = random.choice(facts_positive)
        rule = f"قاعده: همه {animal}ها {verb}"
        fact_text = f"واقعیت: این یک {random.choice(['سنگ', 'درخت', 'ماشین'])} است"
        question = f"آیا می‌توان گفت این {verb}؟"
        text = f"{rule}. {fact_text}. {question}"
        data.append({"text": text, "label": 0, "task_type": "deductive_reasoning"})
        
        # Type 4: If A then B + Not A → Unknown
        cond, cons, _, _ = random.choice(conditional_rules)
        rule = f"قاعده: اگر {cond}، آنگاه {cons}"
        fact = f"واقعیت: {cond} نیست"
        question = f"آیا {cons}؟"
        text = f"{rule}. {fact}. {question}"
        data.append({"text": text, "label": 0, "task_type": "deductive_reasoning"})
    
    # Contradictions
    for _ in range(n - 2 * (n // 3)):
        animal, verb, fact, name, conclusion = random.choice(facts_negative)
        rule = f"قاعده: همه {animal}ها {verb}"
        fact_text = f"واقعیت: {fact}"
        question = f"آیا می‌توان گفت این {conclusion}؟"
        text = f"{rule}. {fact_text}. {question}"
        data.append({"text": text, "label": 0, "task_type": "deductive_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 2: INDUCTIVE REASONING (استدلال استقرایی)
# =============================================================================
def generate_inductive_dataset(n: int) -> pd.DataFrame:
    """
    🧠 INDUCTIVE REASONING
    ======================
    مسیر: مشاهدات متعدد → نتیجه کلی (احتمالی)
    
    مثال:
    - مشاهده ۱: کبوتر پرواز می‌کند
    - مشاهده ۲: گنجشک پرواز می‌کند
    - مشاهده ۳: عقاب پرواز می‌کند
    - نتیجه: "همه پرندگان پرواز می‌کنند" (احتمالاً)
    
    مسیر استدلالی: Observations → Pattern → Generalization
    """
    data = []
    
    # مشاهدات خاص برای استقرا
    observation_sets = [
        # Set 1: Birds fly
        ("کبوتر پرواز می‌کند", "گنجشک پرواز می‌کند", "عقاب پرواز می‌کند", 
         "آیا همه پرندگان پرواز می‌کنند؟"),
        # Set 2: Dogs are loyal
        ("سگ وفادار است", "سگ صاحبش را دوست دارد", "سگ از صاحبش محافظت می‌کند",
         "آیا همه سگ‌ها وفادار هستند؟"),
        # Set 3: Fruits are healthy
        ("سیب سالم است", "پرتقال ویتامین دارد", "انگور آنتی‌اکسیدان دارد",
         "آیا همه میوه‌ها سالم هستند؟"),
        # Set 4: Technology makes life easier
        ("گوشی کارها را آسان می‌کند", "کامپیوتر سریع‌تر کار می‌کند", "اینترنت اطلاعات می‌دهد",
         "آیا همه فناوری‌ها زندگی را آسان می‌کنند؟"),
    ]
    
    # Strong induction (high confidence)
    for _ in range(n // 4):
        obs1, obs2, obs3, question = random.choice(observation_sets)
        text = f"مشاهده ۱: {obs1}. مشاهده ۲: {obs2}. مشاهده ۳: {obs3}. {question}"
        data.append({"text": text, "label": 1, "task_type": "inductive_reasoning"})
    
    # Weak induction (low confidence - only 2 observations)
    for _ in range(n // 4):
        obs1, obs2, _, question = random.choice(observation_sets)
        text = f"مشاهده ۱: {obs1}. مشاهده ۲: {obs2}. {question}"
        data.append({"text": text, "label": 0, "task_type": "inductive_reasoning"})
    
    # Counter-example exists
    for _ in range(n // 4):
        obs1, obs2, obs3, question = random.choice(observation_sets)
        counter = "پنگوئن پرواز نمی‌کند" if "پرندگان" in question else \
                   "گربه وفادار نیست" if "سگ" in question else \
                   "سیگار مضر است" if "میوه" in question else "فناوری قدیمی سخت است"
        text = f"مشاهده ۱: {obs1}. مشاهده ۲: {obs2}. مشاهده ۳: {obs3}. مثال نقض: {counter}. {question}"
        data.append({"text": text, "label": 0, "task_type": "inductive_reasoning"})
    
    # False pattern
    for _ in range(n // 4):
        obs1 = "کبوتر پرواز می‌کند"
        obs2 = "هواپیما پرواز می‌کند"
        obs3 = "مرغ پرواز می‌کند"
        question = "آیا همه چیزهای پرنده پرواز می‌کنند؟"
        text = f"مشاهده ۱: {obs1}. مشاهده ۲: {obs2}. مشاهده ۳: {obs3}. {question}"
        data.append({"text": text, "label": 0, "task_type": "inductive_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 3: ABDUCTIVE REASONING (استدلال ابدکتیو)
# =============================================================================
def generate_abductive_dataset(n: int) -> pd.DataFrame:
    """
    🧠 ABDUCTIVE REASONING
    ======================
    مسیر: پدیده → بهترین توضیح ممکن
    
    مثال:
    - پدیده: زمین خیس است
    - توضیحات ممکن: باران باریده / کسی آب ریخته / لوله ترکیده
    - بهترین توضیح: باران باریده (ساده‌ترین و محتمل‌ترین)
    
    مسیر: Observation → Best Explanation (معمولاً ساده‌ترین)
    """
    data = []
    
    phenomena_explanations = [
        # Phenomenon → [Possible explanations] → Best one
        ("زمین خیس است", ["باران باریده", "کسی آب ریخته", "لوله ترکیده"], "باران باریده"),
        ("آسمان تاریک است", ["شب شده", "طوفان می‌آید", "آتش‌سوزی است"], "شب شده"),
        ("کسی سرفه می‌کند", ["سرماخورده", "آلرژی دارد", "عصبانی است"], "سرماخورده"),
        ("موتور ماشین روشن نمی‌شود", ["باتری خالی است", "بنزین تمام شده", "شمع خراب است"], "بنزین تمام شده"),
        ("صفحه کامپیوتر سیاه است", ["برق نیست", "کامپیوتر خاموش است", "ویروس دارد"], "برق نیست"),
        ("گیاه زرد شده", ["آب کم داده", "آفتاب زیاد", "خاک بد است"], "آب کم داده"),
        ("کسی دیر آمده", ["ترافیک بود", "بیدار نمانده", "دستش درد می‌کند"], "ترافیک بود"),
        ("غذا خوب نیست", ["آشپز بد است", "مواد تازه نیست", "دستور غلط است"], "مواد تازه نیست"),
    ]
    
    # Best explanation (simplest, most likely)
    for _ in range(n // 3):
        phenomenon, explanations, best = random.choice(phenomena_explanations)
        text = f"پدیده: {phenomenon}. توضیحات ممکن: {explanations[0]}، {explanations[1]}، {explanations[2]}. بهترین توضیح کدام است؟"
        data.append({"text": text, "label": 1, "task_type": "abductive_reasoning"})
    
    # Less likely explanation
    for _ in range(n // 3):
        phenomenon, explanations, _ = random.choice(phenomena_explanations)
        text = f"پدیده: {phenomenon}. توضیحات ممکن: {explanations[0]}، {explanations[1]}، {explanations[2]}. آیا {explanations[2]} بهترین توضیح است؟"
        data.append({"text": text, "label": 0, "task_type": "abduction_reasoning"})
    
    # No clear best (equally likely)
    for _ in range(n // 3):
        phenomenon, explanations, _ = random.choice(phenomena_explanations)
        text = f"پدیده: {phenomenon}. توضیح: {explanations[0]}. آاین تنها توضیح ممکن است؟"
        data.append({"text": text, "label": 0, "task_type": "abductive_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 4: ANALOGICAL REASONING (استدلال قیاسی)
# =============================================================================
def generate_analogical_dataset(n: int) -> pd.DataFrame:
    """
    🧠 ANALOGICAL REASONING
    =======================
    مسیر: A:B :: C:? (رابطه بین A و B را پیدا کن، سپس همان رابطه را با C بساز)
    
    مثال:
    - رابطه: قلم مثل کتاب است (برای نوشتن/خواندن)
    - سوال: چاپ مثل چیست؟ (مثل نوشتن → مثل خواندن = کتاب)
    
    مسیر: Source Relation → Map to Target
    """
    data = []
    
    # Analogies: (A, B, relation, C, ?)
    analogies = [
        # Part-whole
        ("دست", "بدن", "بخشی از", "پا", "بخشی از"),
        ("چرخ", "ماشین", "بخشی از", "فرمان", "بخشی از"),
        # Tool-purpose
        ("قلم", "نوشتن", "برای", "چاقو", "برش"),
        ("چکش", "میخ زدن", "برای", " اره", "برش"),
        # Synonym
        ("خوب", "عالی", "مثل", "بد", "فاجعه"),
        ("سریع", "آهسته", "مخالف", "بالا", "پایین"),
        # Cause-effect
        ("آتش", "دود", "تولید می‌کند", "باران", "خیس"),
        # Sequence
        ("صبح", "روز", "قبل از", "ظهر", "روز"),
        # Function
        ("قلب", "پمپاژ خون", "وظیفه", "شش‌ها", "تنفس"),
        # Creator-product
        ("نویسنده", "کتاب", "می‌نویسد", "آهنگساز", "آهنگ"),
    ]
    
    # Direct analogy (positive)
    for _ in range(n // 3):
        a, b, rel, c, _ = random.choice(analogies)
        text = f"رابطه: {a} مثل {b} است (چون {rel}). سوال: چاپ مثل چیست؟ (مثل نوشتن → کتاب)"
        data.append({"text": text, "label": 1, "task_type": "analogical_reasoning"})
    
    # Find the relation
    for _ in range(n // 3):
        a, b, rel, c, answer = random.choice(analogies)
        text = f"{a} نسبت به {b} چه رابطه‌ای دارد؟ {c} نسبت به _______"
        data.append({"text": text, "label": 1, "task_type": "analogical_reasoning"})
    
    # False analogy
    for _ in range(n // 3):
        a, b, rel, c, _ = random.choice(analogies)
        wrong_answer = random.choice(["سنگ", "درخت", "آب", "هوا", "آسمان"])
        text = f"{a} مثل {b} است. {c} مثل {wrong_answer} است. آیا درست است؟"
        data.append({"text": text, "label": 0, "task_type": "analogical_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 5: COUNTERFACTUAL REASONING (استدلال ضد факتی)
# =============================================================================
def generate_counterfactual_dataset(n: int) -> pd.DataFrame:
    """
    🧠 COUNTERFACTUAL REASONING
    ===========================
    مسیر: اگر X (که واقع نشده) بود، چه می‌شد؟
    
    مثال:
    - واقعیت: باران نبارید
    - ضد واقعیت: اگر باران باریده بود، زمین خیس می‌شد
    
    مسیر: Counterfactual Condition → Hypothetical Result
    """
    data = []
    
    counterfactuals = [
        # Past counterfactual
        ("او درس خواند", "اگر درس نمی‌خواند، مردود می‌شد", "مردود شده"),
        ("باران بارید", "اگر باران نمی‌بارید، زمین خشک می‌ماند", "خشک"),
        ("اتوبوس دیر رسید", "اگر زودتر می‌رسید، می‌رسید", "نرسیده"),
        
        # Present counterfactual
        ("او ثروتمند است", "اگر کار نمی‌کرد، فقیر بود", "فقیر"),
        ("هوا سرد است", "اگر تابستان بود، گرم بود", "گرم"),
        
        # Future counterfactual
        ("او امتحان می‌دهد", "اگر بیمار شود، شرکت نمی‌کند", "شرکت نمی‌کند"),
        ("کارخانه تعطیل می‌شود", "اگر سود نکند، کارگران بیکار می‌شوند", "بیکار"),
    ]
    
    # Valid counterfactual
    for _ in range(n // 3):
        fact, cf, result = random.choice(counterfactuals)
        text = f"واقعیت: {fact}. سوال: {cf}؟"
        data.append({"text": text, "label": 1, "task_type": "counterfactual_reasoning"})
    
    # Invalid counterfactual (impossible)
    for _ in range(n // 3):
        fact, cf, result = random.choice(counterfactuals)
        text = f"واقعیت: {fact}. سوال: اگر سنگ پرواز کند، {result}؟"
        data.append({"text": text, "label": 0, "task_type": "counterfactual_reasoning"})
    
    # Counterfactual without proper causation
    for _ in range(n // 3):
        fact, cf, result = random.choice(counterfactuals)
        text = f"واقعیت: {fact}. سوال: اگر ۲+۲=۵ بود، {result}؟"
        data.append({"text": text, "label": 0, "task_type": "counterfactual_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 6: CAUSAL REASONING (استدلال علّی)
# =============================================================================
def generate_causal_dataset(n: int) -> pd.DataFrame:
    """
    🧠 CAUSAL REASONING
    ===================
    مسیر: علت → معلول (چرا؟ چون...)
    
    مثال:
    - چرا گیاه مرد؟ چون آب ندادیم
    - اگر این را بکنی، آن می‌شود
    
    مسیر: Cause → Effect (direct) / Effect → Cause (reverse)
    """
    data = []
    
    causal_pairs = [
        # Direct causation
        ("چون باران بارید، زمین خیس شد", "باران", "زمین خیس"),
        ("چون درس خواندی، قبول شدی", "درس خواندن", "قبول شدن"),
        ("چون باتری تمام شد، موبایل خاموش شد", "باتری تمام", "خاموش شدن"),
        ("چون دیر خوابیدی، خسته هستی", "دیر خوابیدن", "خستگی"),
        ("چون رژیم گرفتی، لاغر شدی", "رژیم", "لاغر شدن"),
        
        # Reverse causation
        ("زمین خیس است. علت چیست؟", "باران", "آب ریختن"),
        ("موتور روشن نمی‌شود. چرا؟", "بنزین", "برق"),
        ("گیاه زرد شده. علت چیست؟", "آب", "نور"),
        
        # No causation
        ("دیروز کبوتر دیدم. امروز باران آمد.", "ارتباطی نیست", "تصادف"),
    ]
    
    # Direct cause-effect
    for _ in range(n // 3):
        statement, cause, effect = random.choice(causal_pairs[:5])
        text = f"{statement}. علت چیست؟"
        data.append({"text": text, "label": 1, "task_type": "causal_reasoning"})
    
    # Reverse (effect to cause)
    for _ in range(n // 3):
        statement, cause, effect = random.choice(causal_pairs[5:8])
        text = f"{statement} چون چیست؟"
        data.append({"text": text, "label": 1, "task_type": "causal_reasoning"})
    
    # No causation
    for _ in range(n // 3):
        statement, cause, effect = random.choice(causal_pairs[8:])
        text = f"{statement}. آیا علت و معلول مرتبط هستند؟"
        data.append({"text": text, "label": 0, "task_type": "causal_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 7: TEMPORAL REASONING (استدلال زمانی)
# =============================================================================
def generate_temporal_dataset(n: int) -> pd.DataFrame:
    """
    🧠 TEMPORAL REASONING
    =====================
    مسیر: روابط زمانی (قبل/بعد/همزمان)
    
    مثال:
    - اول کباب کن، بعد سرو کن
    - وقتی بارید، خیابان خیس شد
    
    مسیر: Time Relations → Sequence/Order
    """
    data = []
    
    temporal_relations = [
        # Before-after
        ("اول صبحانه بخور، بعد کار کن", "صبحانه", "کار", "قبل", 1),
        ("اول درس بخوان، بعد بازی کن", "درس", "بازی", "قبل", 1),
        ("آیا می‌توان همزمان کار کرد و خوابید؟", "کار", "خواب", "همزمان", 0),
        
        # Sequence
        ("یکشنبه قبل از دوشنبه است", "یکشنبه", "دوشنبه", "قبل", 1),
        ("ظهر بعد از صبح است", "ظهر", "صبح", "بعد", 1),
        
        # Duration
        ("فیلم ۲ ساعته است. ۱ ساعت گذشت. آیا تمام شده؟", "1 ساعت", "2 ساعت", "خیر", 0),
        
        # Temporal inference
        ("جلسه ساعت ۳ شروع شد. ساعت ۴ است. آیا تمام شده؟", "3", "4", "شاید", 1),
        
        # Causation in time
        ("اول برق رفت، بعد چراغ خاموش شد", "برق رفتن", "خاموش شدن", "بعد", 1),
    ]
    
    # Valid temporal relations
    for _ in range(n // 3):
        text, _, _, relation, label = random.choice(temporal_relations[:4])
        data.append({"text": text, "label": label, "task_type": "temporal_reasoning"})
    
    # Invalid temporal
    for _ in range(n // 3):
        text, _, _, relation, label = random.choice(temporal_relations[4:7])
        data.append({"text": text, "label": label, "task_type": "temporal_reasoning"})
    
    # Ambiguous
    for _ in range(n // 3):
        text = "آیا می‌توان فهمید چه چیزی اول happened?"
        data.append({"text": text, "label": 0, "task_type": "temporal_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 8: SPATIAL REASONING (استدلال فضایی)
# =============================================================================
def generate_spatial_dataset(n: int) -> pd.DataFrame:
    """
    🧠 SPATIAL REASONING
    ====================
    مسیر: روابط مکانی (بالا/پایین/جلو/ عقب/داخل/بیرون)
    
    مثال:
    - کتاب زیر میز است
    - خانه بین درخت و پارک است
    
    مسیر: Spatial Relations → Location Inference
    """
    data = []
    
    spatial_relations = [
        # Position
        ("کتاب روی میز است", "بالای میز", 1),
        ("توپ زیر تخت است", "پایین تخت", 1),
        ("گربه جلوی در است", "جلوی در", 1),
        ("ماشین پشتی خانه است", "پشت خانه", 1),
        
        # Containment
        ("پول در کیف است", "داخل کیف", 1),
        ("ماهی در آب است", "داخل آب", 1),
        
        # Relative position
        ("بین درخت و رودخانه", "کنار رودخانه", 0),
        ("شمال تهران", "جنوب تهران", 0),
        
        # Navigation
        ("از اینجا به سمت راست برو", "چپ", 0),
    ]
    
    # Correct spatial
    for _ in range(n // 3):
        text, location, _ = random.choice(spatial_relations[:6])
        question = f"کجا؟"
        data.append({"text": f"{text}. {question}", "label": 1, "task_type": "spatial_reasoning"})
    
    # Wrong spatial
    for _ in range(n // 3):
        text, location, _ = random.choice(spatial_relations[6:9])
        question = f"آیا {location} است؟"
        data.append({"text": f"{text}. {question}", "label": 0, "task_type": "spatial_reasoning"})
    
    # Inference needed
    for _ in range(n // 3):
        text = "خانه بین مدرسه و فروشگاه است. مدرسه سمت چپ است. خانه کجاست؟"
        data.append({"text": text, "label": 1, "task_type": "spatial_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 9: QUANTITATIVE REASONING (استدلال کمی)
# =============================================================================
def generate_quantitative_dataset(n: int) -> pd.DataFrame:
    """
    🧠 QUANTITATIVE REASONING
    =========================
    مسیر: عملیات ریاضی (جمع/تفریق/ضرب/تقسیم/مقایسه)
    
    مثال:
    - ۵ سیب دارم، ۳ تا دادم. چند تا مانده؟
    - کدام بزرگ‌تر است؟
    
    مسیر: Numbers → Operations → Result
    """
    data = []
    
    # Arithmetic
    arithmetic = [
        ("۵ + ۳ = ?", 8),
        ("۱۰ - ۴ = ?", 6),
        ("۳ × ۷ = ?", 21),
        ("۲۰ ÷ ۴ = ?", 5),
        ("۱۵ + ۲۵ - ۱۰ = ?", 30),
    ]
    
    # Comparison
    comparisons = [
        ("کدام بزرگ‌تر است: ۷ یا ۹؟", "۹", 1),
        ("کدام کوچک‌تر است: ۱۵ یا ۱۲؟", "۱۲", 1),
        ("آیا ۱۰۰ بزرگ‌تر از ۵۰ است؟", "بله", 1),
    ]
    
    # Word problems
    word_problems = [
        ("علی ۵ سیب دارد. ۳ سیب به حسن می‌دهد. چند سیب می‌ماند؟", 2),
        ("یک کارگر ۸ ساعت کار می‌کند. اگر ۳ ساعت استراحت کند، چند ساعت کار کرده؟", 5),
        ("قیمت کتاب ۱۰ هزار تومان است. با ۵۰ هزار تومان چند کتاب می‌توان خرید؟", 5),
    ]
    
    # Valid calculations
    for _ in range(n // 3):
        q, ans = random.choice(arithmetic)
        text = f"سوال: {q}"
        data.append({"text": text, "label": 1, "task_type": "quantitative_reasoning"})
    
    # Valid comparisons
    for _ in range(n // 3):
        q, ans, _ = random.choice(comparisons)
        text = f"سوال: {q}"
        data.append({"text": text, "label": 1, "task_type": "quantitative_reasoning"})
    
    # Word problems
    for _ in range(n // 3):
        q, ans = random.choice(word_problems)
        text = f"سوال: {q}"
        data.append({"text": text, "label": 1, "task_type": "quantitative_reasoning"})
    
    # Wrong answer
    for _ in range(n // 6):
        q, ans = random.choice(arithmetic)
        wrong = ans + random.randint(1, 5)
        text = f"سوال: {q} (جواب: {wrong})"
        data.append({"text": text, "label": 0, "task_type": "quantitative_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# TASK 10: SOCIAL REASONING (استدلال اجتماعی)
# =============================================================================
def generate_social_dataset(n: int) -> pd.DataFrame:
    """
    🧠 SOCIAL REASONING
    ===================
    مسیر: درک نیت، باور، خواسته، احساس افراد
    
    مثال:
    - علی می‌داند که کتاب روی میز است
    - حسن می‌خواهد آب بخورد
    - مریم فکر می‌کند امتحان سخت است
    
    مسیر: Observe Behavior → Infer Mental State
    """
    data = []
    
    social_scenarios = [
        # Belief (باور)
        ("علی فکر می‌کند امروز باران می‌آید", "باور", "باران می‌آید", 1),
        ("سارا می‌داند که کلید در جیبش است", "دانستن", "کلید در جیب", 1),
        ("راوی فکر می‌کند زمین صاف است", "باور اشتباه", "زمین گرد", 0),
        
        # Desire (میل)
        ("حسن می‌خواهد آب بخورد", "میل", "آب", 1),
        ("مریم آرزوی پرواز دارد", "آرزو", "پرواز", 1),
        ("کسی گرسنه نیست اما می‌خواهد غذا بخورد", "میل", "غذا", 0),
        
        # Intention (قصد)
        ("علی قصد دارد فردا به مدرسه برود", "قصد", "رفتن به مدرسه", 1),
        ("او می‌خواهد کمک کند اما نمی‌تواند", "قصد", "کمک", 1),
        
        # Emotion (احساس)
        ("چون باران می‌بارد، ناراحت است", "احساس", "ناراحتی", 1),
        ("چون کادو گرفت، خوشحال است", "احساس", "خوشحالی", 1),
        
        # Deception (فریب)
        ("می‌گوید دوستت دارد اما قصد دارد به تو خیانت کند", "نیت واقعی", "خیانت", 0),
    ]
    
    # Valid social inference
    for _ in range(n // 3):
        scenario, mental_type, content, _ = random.choice(social_scenarios[:6])
        text = f"{scenario}. نوع حالت ذهنی چیست؟"
        data.append({"text": text, "label": 1, "task_type": "social_reasoning"})
    
    # Intent vs belief
    for _ in range(n // 3):
        scenario, mental_type, content, _ = random.choice(social_scenarios[6:9])
        text = f"{scenario}. آیا این یک {mental_type} است؟"
        data.append({"text": text, "label": 1, "task_type": "social_reasoning"})
    
    # False inference
    for _ in range(n // 3):
        scenario, mental_type, content, _ = random.choice(social_scenarios[9:])
        text = f"{scenario}. آیا می‌توان گفت این {mental_type} است؟"
        data.append({"text": text, "label": 0, "task_type": "social_reasoning"})
    
    return pd.DataFrame(data)


# =============================================================================
# MAIN GENERATOR
# =============================================================================
# ... (کد قبلی شما شامل توابع generate_deductive_dataset تا generate_social_dataset بدون تغییر باقی می‌ماند) ...

# =============================================================================
# MAIN GENERATOR (ادامه بخش قطع شده)
# =============================================================================

def generate_reasoning_datasets(output_dir: Path, samples_per_task: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Generate all 10 reasoning task datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generators = {
        "task01_deductive": generate_deductive_dataset,
        "task02_inductive": generate_inductive_dataset,
        "task03_abductive": generate_abductive_dataset,
        "task04_analogical": generate_analogical_dataset,
        "task05_counterfactual": generate_counterfactual_dataset,
        "task06_causal": generate_causal_dataset,
        "task07_temporal": generate_temporal_dataset,
        "task08_spatial": generate_spatial_dataset,
        "task09_quantitative": generate_quantitative_dataset,
        "task10_social": generate_social_dataset,
    }
    
    datasets = {}
    
    print("=" * 75)
    print("🎯 ADVANCED REASONING DATASET GENERATOR")
    print("=" * 75)
    print(f"📊 Samples per task: {samples_per_task}")
    print(f"📁 Output: {output_dir}")
    print("-" * 75)
    
    for task_name, generator in generators.items():
        print(f"\n🔄 Generating: {task_name}...")
        
        df = generator(samples_per_task)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        file_path = output_dir / f"{task_name}.csv"
        df.to_csv(file_path, index=False)
        
        datasets[task_name] = df
        
        pos = df['label'].sum()
        neg = len(df) - pos
        total = len(df)
        
        print(f"   ✅ Saved to: {file_path.name}")
        print(f"   📈 Total: {total} | Positive: {pos} | Negative: {neg}")
        print(f"   📉 Balance Ratio: {pos/total:.2%} vs {neg/total:.2%}")

    # Combine all datasets into one master file
    print("\n" + "=" * 75)
    print("🔗 Combining all tasks into a single dataset...")
    
    master_df = pd.concat(datasets.values(), ignore_index=True)
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    master_file = output_dir / "all_reasoning_tasks_combined.csv"
    master_df.to_csv(master_file, index=False)
    
    print(f"✅ Master dataset saved: {master_file.name}")
    print(f"📊 Total combined samples: {len(master_df)}")
    print(f"🏷️ Task distribution:\n{master_df['task_type'].value_counts()}")
    print("=" * 75)
    print("✨ Generation Complete!")
    print("=" * 75)
    
    return datasets


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # تنظیم مسیر خروجی (می‌توانید آن را تغییر دهید)
    OUTPUT_PATH = Path("./datasets")
    
    try:
        # اجرای تابع اصلی با مقدار پیش‌فرض ۱۰۰۰ نمونه برای هر وظیفه
        # اگر خطای حافظه داشتید، عدد را کاهش دهید (مثلاً ۵۰۰)
        results = generate_reasoning_datasets(OUTPUT_PATH, samples_per_task=1000)
        
        print("\n💡 نکته فنی:")
        print("- فایل‌های جداگانه برای هر نوع استدلال در پوشه 'datasets' ایجاد شدند.")
        print("- یک فایل ترکیبی ('all_reasoning_tasks_combined.csv') نیز تولید شد.")
        print("- برای آموزش مدل BERT یا سایر مدل‌ها، از فایل ترکیبی استفاده کنید.")
        
    except Exception as e:
        print(f"\n❌ Error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)