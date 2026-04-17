import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
# ✅ حذف BaseModelOutput... از اینجا برای جلوگیری از خطا در نسخه‌های قدیمی
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# راه حل سازگاری برای نسخه‌های قدیمی Transformers
# ---------------------------------------------------------
try:
    # تلاش برای دریافت کلاس از خروجی BERT (روش ایمن)
    from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions
except ImportError:
    # اگر در ماژول modeling_bert هم نبود، یک نسخه ساده تعریف می‌کنیم
    # این نسخه فقط ساختار داده‌ای مورد نیاز را شبیه‌سازی می‌کند
    class BaseModelOutputWithPoolingAndCrossAttentions(nn.Module):
        def __init__(self, last_hidden_state=None, pooler_output=None, hidden_states=None, attentions=None, cross_attentions=None):
            super().__init__()
            self.last_hidden_state = last_hidden_state
            self.pooler_output = pooler_output
            self.hidden_states = hidden_states
            self.attentions = attentions
            self.cross_attentions = cross_attentions

        def __getitem__(self, key):
            # برای سازگاری با دسترسی مثل دیکشنری در صورت نیاز
            if key == 'last_hidden_state': return self.last_hidden_state
            if key == 'pooler_output': return self.pooler_output
            raise KeyError(key)



warnings.filterwarnings('ignore')
class DatasetConfig:
    def __init__(self, name, path, task_type, num_choices, num_labels):
        self.name = name
        self.path = path
        self.task_type = task_type
        self.num_choices = num_choices
        self.num_labels = num_labels

def discover_datasets(config=None) -> list:
    dataset_paths = {
        # "ss2": "./data/ss2.csv",
        # "hellaswag": "./data/reasoning/hellaswag/processed/hellaswag_train.csv",
        # "data_custom": "./data/data_custom.csv",
        "arc_challenge": "./data/reasoning/arc_challenge/processed/arc_challenge_train.csv",
        "arc_agi": "./data/reasoning/arc_agi/processed/arc_agi_train.csv",
    }
    
    datasets = []
    for name, path in dataset_paths.items():
        if not Path(path).exists():
            print(f"❌ {name}: File not found at {path}")
            continue
        try:
            df_sample = pd.read_csv(path, nrows=5)
            is_multichoice = False
            num_choices = 0
            final_num_labels = 2
            final_task_type = 'single'
            
            has_choices_json = 'choices_json' in df_sample.columns
            has_gold_answer = 'gold_answer_json' in df_sample.columns or 'answer_json' in df_sample.columns
            
            if has_choices_json and has_gold_answer:
                first_val = df_sample['choices_json'].iloc[0]
                if isinstance(first_val, str):
                    try:
                        choices_list = json.loads(first_val)
                        if isinstance(choices_list, list) and len(choices_list) >= 2:
                            is_multichoice = True
                            num_choices = len(choices_list)
                            final_num_labels = num_choices
                            final_task_type = 'mc'
                            print(f"✅ Detected {name}: Multi-Choice ({num_choices} options from JSON)")
                    except json.JSONDecodeError:
                        continue
                elif isinstance(first_val, list) and len(first_val) >= 2:
                    is_multichoice = True
                    num_choices = len(first_val)
                    final_num_labels = num_choices
                    final_task_type = 'mc'
                    print(f"✅ Detected {name}: Multi-Choice ({num_choices} options directly)")
            elif 'choice_a' in df_sample.columns and 'answer_label' in df_sample.columns:
                choice_cols = [col for col in df_sample.columns if col.startswith('choice_')]
                if len(choice_cols) >= 2:
                    is_multichoice = True
                    num_choices = len(choice_cols)
                    final_num_labels = num_choices
                    final_task_type = 'mc'
                    print(f"✅ Detected {name}: Multi-Choice ({num_choices} options - ARC Style)")
            elif 'endings' in df_sample.columns:
                 is_multichoice = True
                 final_task_type = 'mc'
                 print(f"✅ Detected {name}: Multi-Choice (Endings Column)")
            elif 'label' in df_sample.columns:
                final_task_type = 'single'
                print(f"✅ Detected {name}: Single Label")
            else:
                print(f"⚠️ {name}: No clear labels found.")
                continue
            
            datasets.append(DatasetConfig(name=name, path=path, task_type=final_task_type,
                num_choices=num_choices if is_multichoice else 0, num_labels=final_num_labels))
        except Exception as e:
            print(f"⚠️ Error analyzing {name}: {e}")
            continue
            
    return datasets

# ---------------------------------------------------------
# 2. لایه SVD اختصاصی (اصلاح شده برای گرادیان‌پذیری محدود / پیش‌پردازش)
# ---------------------------------------------------------
class DirectSVDEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # ✅ 1. لایه نرمالیزاسیون قبل از SVD
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        # ✅ 2. پردازشگر SVD
        self.svd_layer = SVDSingleHead(hidden_size=self.hidden_size, svd_rank=20) # رتبه را اینجا هم تنظیم کن
        
        # ✅ 3. لایه فید فورارد (FFN) دقیقاً مثل BERT
        self.intermediate = nn.Linear(self.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # ✅ 4. لایه نرمالیزاسیون بعد از FFN
        self.output_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # ذخیره ورودی اصلی برای اتصال باقیمانده (Residual)
        residual = hidden_states
        
        # --- بخش اول: SVD ---
        # نرمالیزاسیون ورودی
        normed_input = self.input_layernorm(hidden_states)
        
        # پردازش SVD
        svd_output = self.svd_layer(normed_input)
        
        # اتصال باقیمانده
        hidden_states = residual + svd_output
        
        # --- بخش دوم: Feed Forward Network (مثل BERT) ---
        residual = hidden_states
        
        # نرمالیزاسیون ورودی به FFN
        normed_ffn = self.output_layernorm(hidden_states)
        
        # لایه اول FFN
        intermediate_output = self.intermediate(normed_ffn)
        # فعال‌سازی GeLU
        intermediate_output = torch.nn.functional.gelu(intermediate_output)
        
        # لایه دوم FFN
        ff_output = self.output_dense(intermediate_output)
        ff_output = self.dropout(ff_output)
        
        # اتصال باقیمانده نهایی
        hidden_states = residual + ff_output
        
        # بازگشت خروجی
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (hidden_states,) # یا ذخیره حالت قبلی اگر نیاز باشد
        
        return outputs[0]


# ---------------------------------------------------------
# کلاس کمکی SVD (همان کد قبلی با اصلاحات)
# ---------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------
# کلاس کمکی SVD (با نرمال‌سازی ورودی)
# ---------------------------------------------------------
class SVDSingleHead(nn.Module):
    def __init__(self, hidden_size, svd_rank=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.svd_rank = svd_rank
        self.register_buffer('pos_encoding', None)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # مدیریت پوزیشنال انکدینگ
        if self.pos_encoding is None or self.pos_encoding.shape[0] != seq_len:
            pe = torch.zeros(seq_len, self.hidden_size, device=hidden_states.device)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-np.log(10000.0) / self.hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_encoding = pe.unsqueeze(0)
        
        x_with_pos = hidden_states + self.pos_encoding[:, :seq_len, :]
        
        # نرمال‌سازی ورودی قبل از SVD (برای پایداری عددی)
        X_flat = torch.nn.functional.normalize(x_with_pos.reshape(-1, self.hidden_size), p=2, dim=1)
        
        try:
            U, s, Vh = torch.linalg.svd(X_flat, full_matrices=False)
            
            k = min(self.svd_rank, len(s))
            top_s = s[:k]
            top_Vh = Vh[:k, :]
            
            # Softmax روی مقادیر ویژه برای توزیع وزن
            s_max = top_s.max()
            exp_s = torch.exp(top_s - s_max)
            softmax_s = exp_s / exp_s.sum()
            
            weighted_U = U[:, :k] * softmax_s.view(1, -1)
            output_flat = torch.matmul(weighted_U, top_Vh)
            
            output = output_flat.reshape(batch_size, seq_len, self.hidden_size)
            return output
            
        except RuntimeError as e:
            print(f"⚠️ SVD failed in SVDSingleHead: {e}. Returning normalized input.")
            return x_with_pos


# ---------------------------------------------------------
# کلاس لایه هیبریدی (جایگزین Attention)
# ---------------------------------------------------------
class DirectSVDEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # ✅ 1. لایه نرمالیزاسیون قبل از SVD
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        # ✅ 2. پردازشگر SVD
        self.svd_layer = SVDSingleHead(hidden_size=self.hidden_size, svd_rank=20)
        
        # ✅ 3. لایه فید فورارد (FFN) دقیقاً مثل BERT
        self.intermediate = nn.Linear(self.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # ✅ 4. لایه نرمالیزاسیون بعد از FFN
        self.output_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # ذخیره ورودی اصلی برای اتصال باقیمانده (Residual)
        residual = hidden_states
        
        # --- بخش اول: SVD ---
        # نرمالیزاسیون ورودی
        normed_input = self.input_layernorm(hidden_states)
        
        # پردازش SVD
        svd_output = self.svd_layer(normed_input)
        
        # اتصال باقیمانده
        hidden_states = residual + svd_output
        
        # --- بخش دوم: Feed Forward Network (مثل BERT) ---
        residual = hidden_states
        
        # نرمالیزاسیون ورودی به FFN
        normed_ffn = self.output_layernorm(hidden_states)
        
        # لایه اول FFN
        intermediate_output = self.intermediate(normed_ffn)
        # فعال‌سازی GeLU
        intermediate_output = torch.nn.functional.gelu(intermediate_output)
        
        # لایه دوم FFN
        ff_output = self.output_dense(intermediate_output)
        ff_output = self.dropout(ff_output)
        
        # اتصال باقیمانده نهایی
        hidden_states = residual + ff_output
        
        # بازگشت خروجی
        # توجه: چون ما Attention نداریم، output_attentions و cross_attentions را نادیده می‌گیریم
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (hidden_states,) 
        
        return outputs[0]
        
        
class HybridBertWithDirectSVDBert(BertModel):
    def __init__(self, config):
        super().__init__(config)
        # نگه داشتن Embeddings و Pooler از مدل پایه
        # جایگزینی encoder با لایه‌های هیبریدی
        self.encoder = nn.ModuleList([
            DirectSVDEncoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids=None,
        attention_mask=None, # این پارامتر را نگه می‌داریم اما به embeddings نمی‌فرستیم
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # ✅ 1. فراخوانی Embeddings (اصلاح شده برای نسخه قدیمی)
        # توجه: در نسخه‌های قدیمی، attention_mask اینجا استفاده نمی‌شود.
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
            # attention_mask حذف شد تا خطا ندهد
        )
        
        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # ✅ 2. اجرای لایه‌های هیبریدی
        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # فراخوانی لایه SVD + FFN
            # توجه: ما فقط hidden_states را می‌فرستیم
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # ✅ 3. محاسبه Pooler Output
        pooled_output = None
        if hasattr(self, 'pooler') and self.pooler is not None:
            pooled_output = self.pooler(hidden_states)
            pooled_output = torch.tanh(pooled_output)
        
        # ✅ 4. بازگشت
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )
        
        
class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        return hidden_states

# ---------------------------------------------------------
# 3. بارگذاری داده‌ها (اصلاح شده)
# ---------------------------------------------------------
def load_dataset_from_config(dataset_config, tokenizer, max_length=128):
    print(f"\n📂 Loading {dataset_config.name}...")
    df = pd.read_csv(dataset_config.path)
    
    # تشخیص ستون‌ها
    text_col = 'text' if 'text' in df.columns else ('question' if 'question' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else 'gold_answer'
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    # تبدیل به توکن‌ها (بدون ساخت تانسور)
    data_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}
    
    for i, text in enumerate(texts):
        encoding = tokenizer(str(text), truncation=True, padding="max_length", max_length=max_length)
        # ذخیره لیست‌های عادی (نه تانسور)
        data_dict['input_ids'].append(encoding['input_ids'])
        data_dict['attention_mask'].append(encoding['attention_mask'])
        data_dict['labels'].append(int(labels[i]))
        
    ds = Dataset.from_dict(data_dict)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print(f"✅ Loaded {len(ds)} samples for {dataset_config.name}")
    return ds

# ---------------------------------------------------------
# 4. متریک‌ها
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')
    }

# ---------------------------------------------------------
# 5. مدل هیبریدی نهایی
# ---------------------------------------------------------
class HybridBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = HybridBertWithDirectSVDBert(config)
        self.classifier = ClassificationHead(config)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # خروجی از BERT هیبریدی
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # گرفتن pooled output یا last hidden state
        # معمولا از [CLS] token استفاده می‌شود
        cls_output = outputs.last_hidden_state[:, 0, :] 
        
        # طبقه‌بندی
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.dense.out_features), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

# ---------------------------------------------------------
# 6. اجرای اصلی
# ---------------------------------------------------------
# ... (وارد کردن کتابخانه‌ها همانند قبل) ...
# ... (وارد کردن کتابخانه‌ها همانند قبل) ...
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
# فرض بر این است که توابع discover_datasets, load_dataset_from_config, compute_metrics, 
# و کلاس HybridBertForSequenceClassification در فایل‌های دیگر تعریف شده‌اند.

def main():
    # 1. تنظیم دستگاه (Device)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs("results", exist_ok=True)
    print(f"🚀 Device: {device}")
    
    # 2. کشف دیتاست‌ها
    configs = discover_datasets()
    if not configs:
        print("❌ No valid datasets found. Please check your ./data directory.")
        return
    
    selected_config = configs[0]
    print(f"\n🎯 Selected Dataset for Testing: {selected_config.name}")
    
    # 3. بارگذاری توکنایزر
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 4. بارگذاری دیتاست و تقسیم آن
    train_test_split = load_dataset_from_config(selected_config, tokenizer)
    split = train_test_split.train_test_split(test_size=0.2, seed=42)
    train_dataset = split['train']
    test_dataset = split['test']
    
    # ---------------------------------------------------------
    # تابع کمکی برای ساخت TrainingArguments سازگار با نسخه‌های مختلف
    # ---------------------------------------------------------
    def get_training_args(output_dir, is_hybrid=False):
        try:
            args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                num_train_epochs=3,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                logging_steps=50,
                report_to="none",
                disable_tqdm=True
            )
            return args
        except TypeError as e:
            if "evaluation_strategy" in str(e):
                print(f"⚠️ Detected old transformers version. Switching to legacy parameters...")
                args = TrainingArguments(
                    output_dir=output_dir,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=32,
                    num_train_epochs=1,
                    weight_decay=0.01,
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    logging_steps=50,
                    report_to="none",
                    disable_tqdm=True
                )
                return args
            else:
                raise e

    # ---------------------------------------------------------
    # 5. آموزش مدل استاندارد (Baseline)
    # ---------------------------------------------------------
    print("\n🔥 Training Standard BERT...")
    
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=selected_config.num_labels
    )
    base_model.to(device)
    
    training_args = get_training_args("./results/bert_standard")
    
    trainer_base = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer_base.train()
    eval_results_std = trainer_base.evaluate()
    best_acc_standard = eval_results_std.get("eval_accuracy", 0.0)
    print(f"✅ Standard Model Accuracy: {best_acc_standard:.4f}")

    # --- شروع بخش هیبریدی ---
    print("\n🧪 Building Hybrid SVD Model...")
    
    hybrid_config = BertConfig.from_pretrained("bert-base-uncased")
    hybrid_config.num_labels = selected_config.num_labels
    
    hybrid_model = HybridBertForSequenceClassification(hybrid_config)
    hybrid_model.to(device)
    
    # 2. کپی وزن‌ها (مدیریت دقیق‌تر برای حذف هشدارهای UNEXPECTED غیرضروری)
    try:
        temp_base = BertModel.from_pretrained("bert-base-uncased")
        
        # لیست کلیدهایی که نباید بارگذاری شوند (چون در مدل هیبریدی وجود ندارند یا متفاوت هستند)
        # این کار باعث می‌شود هشداری درباره کلیدهای اضافی (Unexpected) نگیریم
        ignore_keys = ['cls.predictions', 'cls.seq_relationship']
        
        state_dict = temp_base.state_dict()
        
        # بارگذاری فقط بخش‌های مشترک (embeddings, encoder layers, pooler)
        missing_keys, unexpected_keys = hybrid_model.bert.load_state_dict(state_dict, strict=False)
        
        # نمایش جزئیات برای دیباگ (اختیاری)
        if missing_keys or unexpected_keys:
            print(f"ℹ️ Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            # اگر نیاز بود می‌توانیم نام کلیدهای خاص را چاپ کنیم، اما فعلا فقط تعداد کافی است.
            
        del temp_base
        print("✅ Copied weights from Base BERT to Hybrid Model successfully.")
        
    except Exception as e:
        print(f"⚠️ Warning copying weights: {e}")
        import traceback
        traceback.print_exc()
        
    # 3. تنظیمات آموزش هیبریدی
    training_args_hybrid = get_training_args("./results/hybrid_svd", is_hybrid=True)
    
    trainer_hybrid = Trainer(
        model=hybrid_model,
        args=training_args_hybrid,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # ---------------------------------------------------------
    # 4. تست بدون آموزش (Zero-shot) - اصلاح شده برای رفع خطای Shape
    # ---------------------------------------------------------
    print("\n🚫 Testing Hybrid Model (No Train)...")
    hybrid_model.eval()
    
    all_predictions, all_true_labels = [], []
    
    with torch.no_grad():
        # نکته حیاتی: در دیتاست‌های Hugging Face، حلقه مستقیم ممکن است نمونه‌های تک‌بعدی برگرداند.
        # ما باید اطمینان حاصل کنیم که input_ids شکل (batch_size, seq_len) دارد.
        # راه حل امن: استفاده از DataCollator برای هر بچ کوچک یا افزودن بعد Batch.
        
        # روش پیشنهادی: جمع‌آوری نمونه‌ها و سپس ساخت بچ، یا افزودن بعد در هر تکرار.
        # اینجا روش افزودن بعد (unsqueeze) را برای سادگی و سرعت در تست تک‌نمونه‌ای انجام می‌دهیم.
        
        for item in test_dataset:
            # گرفتن ورودی‌ها
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            # تبدیل به تانسور و انتقال به دستگاه
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            labels = torch.tensor(labels).to(device)
            
            # ✅ اصلاح خطا: اگر شکل تانسور فقط یک بعدی است (تک نمونه)، بعد Batch اضافه کن
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            # بررسی نهایی شکل (باید حداقل 2 بعد باشد: [batch, seq])
            if input_ids.dim() < 2:
                raise ValueError(f"Input shape is invalid after unsqueeze: {input_ids.shape}")

            outputs = hybrid_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # استخراج پیش‌بینی‌ها
            logits = outputs["logits"]
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
    acc_hybrid_no_train = accuracy_score(all_true_labels, all_predictions)
    print(f"🚫 Hybrid Model (No Train) Accuracy: {acc_hybrid_no_train:.4f}")
    
    # ---------------------------------------------------------
    # 5. آموزش مدل هیبریدی
    # ---------------------------------------------------------
    print("\n🔥 Training Hybrid SVD Model...")
    try:
        trainer_hybrid.train()
        
        print("✅ Hybrid training finished. Evaluating...")
        eval_results_hyb = trainer_hybrid.evaluate()
        best_acc_hybrid = eval_results_hyb.get("eval_accuracy", 0.0)
        print(f"✅ Hybrid Model (After Training) Accuracy: {best_acc_hybrid:.4f}")
        
        print(f"\n📊 Comparison:")
        print(f"Standard BERT: {best_acc_standard:.4f}")
        print(f"Hybrid SVD:    {best_acc_hybrid:.4f}")
        diff = best_acc_hybrid - best_acc_standard
        print(f"Difference:    {diff:+.4f}")
        
        if diff > 0:
            print(f"✨ The Hybrid model improved performance by {diff*100:.2f}%!")
        else:
            print(f"⚠️ The Hybrid model performed slightly worse or equal.")
        
    except Exception as e:
        print(f"❌ Error during Hybrid training: {e}")
        import traceback
        traceback.print_exc()
        best_acc_hybrid = 0.0

if __name__ == "__main__":
    main()