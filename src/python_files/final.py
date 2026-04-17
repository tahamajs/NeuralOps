import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import logging
import argparse
import torch.nn as nn
import gc
import warnings
from tqdm import tqdm
import time
import psutil
import random
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION CLASSES ====================
@dataclass
class DatasetConfig:
    name: str
    path: str
    task_type: str  # 'mc' or 'single'
    num_choices: int = 0
    num_labels: int = 2

@dataclass
class ExperimentConfig:
    base_models: Dict[str, str] = field(default_factory=lambda: {
        "bert": "bert-base-uncased",
    })
    
    svd_ranks: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    datasets: List[DatasetConfig] = field(default_factory=list)
    
    min_samples: int = 10
    max_samples: int = 200
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    val_split: float = 0.2
    seed: int = 42
    gradient_accumulation_steps: int = 1
    save_checkpoint_interval: int = 10
    output_dir: str = "./results"
    plots_dir: str = "./plots/comprehensive"
    log_dir: str = "./logs"
    selected_models: List[str] = field(default_factory=list)
    selected_datasets: List[str] = field(default_factory=list)
    selected_ranks: List[int] = field(default_factory=list)

def setup_experiment(config: ExperimentConfig) -> Tuple[torch.device, logging.Logger]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    
    print(f"⚡ Using device: {device}")
    
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.log_dir) / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return device, logger
def compute_loss(outputs, labels, config):
    return outputs.loss

# ==================== DATASET DISCOVERY ====================
def discover_datasets(config: ExperimentConfig) -> List[DatasetConfig]:
    dataset_paths = {
        # "ss2": "./data/ss2.csv",
        # "hellaswag": "./data/reasoning/hellaswag/processed/hellaswag_train.csv",
        "data_custom": "./data/data_custom.csv",
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

# ==================== DATASET CLASSES ====================
class MultiChoiceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        has_choices_json = 'choices_json' in df.columns
        has_endings = 'endings' in df.columns
        has_gold_answer = 'gold_answer_json' in df.columns or 'answer_json' in df.columns
        
        valid_count = 0
        invalid_count = 0
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            question = ""
            choices = []
            label = 0
            
            if 'prompt' in row: question = str(row['prompt'])
            elif 'ctx' in row: question = str(row['ctx'])
            else: 
                parts = [str(v) for v in row[['ctx', 'ctx_a', 'ctx_b']].dropna().values]
                question = " ".join(parts)
            
            if has_choices_json:
                try: choices_str = str(row['choices_json']); choices = json.loads(choices_str)
                except: pass
            elif has_endings:
                try: endings_str = str(row['endings']); choices = json.loads(endings_str)
                except: pass
            
            if has_gold_answer:
                if 'gold_answer_json' in row:
                    ans_json = json.loads(str(row['gold_answer_json']))
                    if 'choice_index' in ans_json: label = int(ans_json['choice_index'])
                elif 'answer_json' in row:
                    ans_json = json.loads(str(row['answer_json']))
                    if 'choice_index' in ans_json: label = int(ans_json['choice_index'])
            elif 'label' in row:
                label = int(row['label'])
            
            if len(choices) >= 2 and 0 <= label < len(choices):
                self.samples.append({'question': question, 'choices': choices, 'label': label})
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"📊 Dataset Filtered ({len(df)} total): {valid_count} valid kept, {invalid_count} filtered.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        choices = sample['choices']
        label = sample['label']
        
        encoded_choices = []
        for choice in choices:
            text = f"{sample['question']} {choice}"
            enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            encoded_choices.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })
        
        input_ids = torch.stack([ec['input_ids'] for ec in encoded_choices])
        attention_mask = torch.stack([ec['attention_mask'] for ec in encoded_choices])
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label, dtype=torch.long)}

class SingleLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            text = str(row.get('question', row.get('prompt', '')))
            label = 0 
            self.texts.append(text)
            self.labels.append(label)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==================== SV-COMPRESSED BERT MODEL ====================
# ==================== SV-COMPRESSED BERT MODEL (FIXED EMBEDDING COPY) ====================
class CompressedBertModel(nn.Module):
    def __init__(self, model_path: str, rank: int, num_labels: int):
        super().__init__()
        config = BertConfig.from_pretrained(model_path)
        self.config = config
        self.hidden_size = config.hidden_size # 768
        self.rank = rank
        
        # بارگذاری اولیه برای دسترسی به وزن‌ها
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2) 
        base_bert = base_model.bert
        
        # --- 1. فشرده‌سازی Embeddings (Word & Position) ---
        
        # Word Embeddings
        # W_word shape: [30522, 768]
        W_word = base_bert.embeddings.word_embeddings.weight.data.clone()
        U_w, S_w, Vh_w = torch.linalg.svd(W_word, full_matrices=False)
        
        # بازسازی با رنک کاهش یافته: [30522, rank]
        # نکته مهم: نیازی به .t() نیست. U[:, :rank] * S[:rank] مستقیماً شکل [30522, rank] می‌دهد.
        W_compressed_word = U_w[:, :rank] * S_w[:rank].unsqueeze(0)
        
        # ساخت لایه Embedding جدید با بعد خروجی برابر با rank
        self.word_embeddings = nn.Embedding(base_bert.embeddings.word_embeddings.num_embeddings, rank)
        
        # کپی وزن‌ها (اکنون شکل‌ها کاملاً مطابقت دارند: [30522, rank])
        with torch.no_grad():
            self.word_embeddings.weight.copy_(W_compressed_word) # بدون .t()
        
        # Position Embeddings
        W_pos = base_bert.embeddings.position_embeddings.weight.data.clone()
        U_p, S_p, Vh_p = torch.linalg.svd(W_pos, full_matrices=False)
        W_compressed_pos = U_p[:, :rank] * S_p[:rank].unsqueeze(0)
        
        self.position_embeddings = nn.Embedding(base_bert.embeddings.position_embeddings.num_embeddings, rank)
        with torch.no_grad():
            self.position_embeddings.weight.copy_(W_compressed_pos) # بدون .t()
        
        # Token Type Embeddings (اگر وجود دارد)
        # Token Type Embeddings (اگر وجود دارد)
        if hasattr(base_bert.embeddings, 'token_type_embeddings'):
            W_tt = base_bert.embeddings.token_type_embeddings.weight.data.clone()
            
            # نکته مهم: اگر تعداد توکن تایپ‌ها (مثلاً 2) کمتر از رنک درخواستی (مثلاً 8) باشد،
            # SVD نمی‌تواند رنک 8 تولید کند. پس باید رنک را به حداقل تعداد سطرها محدود کنیم.
            actual_rank = min(rank, W_tt.shape[0])
            
            U_tt, S_tt, Vh_tt = torch.linalg.svd(W_tt, full_matrices=False)
            # محاسبه فشرده شده با رنک واقعی
            W_compressed_tt = U_tt[:, :actual_rank] * S_tt[:actual_rank].unsqueeze(0)
            
            # ساخت لایه جدید با بعد خروجی برابر با رنک واقعی (نه رنک اصلی)
            self.token_type_embeddings = nn.Embedding(base_bert.embeddings.token_type_embeddings.num_embeddings, actual_rank)
            
            with torch.no_grad():
                self.token_type_embeddings.weight.copy_(W_compressed_tt)
                
            # ذخیره رنک واقعی برای استفاده در forward (اختیاری)
            self.tt_rank = actual_rank
        else:
            self.token_type_embeddings = None
            self.tt_rank = 0
        # --- 2. ساخت لایه‌های Encoder فشرده شده ---
        # --- 2. ساخت لایه‌های Encoder فشرده شده ---
        self.encoder_layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = BaseLayerCompressed(rank, config) 
            self.encoder_layers.append(layer)
        
        # ✅ اصلاح ۱: نرمالایز نهایی باید با بعد 'rank' باشد
        self.final_layer_norm = nn.LayerNorm(rank) 
        
        
        # ✅ اصلاح ۲ حیاتی: کلاسیفایر باید ورودی 'rank' بگیرد، نه 'hidden_size'
        # اینجا دیگر نباید از self.hidden_size استفاده کنیم چون مدل کاملاً فشرده است.
        self.classifier = nn.Linear(rank, num_labels) 

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # دریافت Embeddings فشرده شده
        word_embeds = self.word_embeddings(input_ids) # [B, L, Rank]
        
        # تولید position_ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) # [B, L]
        
        position_embeds = self.position_embeddings(position_ids) # [B, L, Rank]
        
        embeddings = word_embeds + position_embeds
        
        # حذف Token Type Embeddings برای جلوگیری از خطا
        
        # ارسال به لایه‌های انکودر
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, attention_mask)[0]
        
        embeddings = self.final_layer_norm(embeddings)
        
        # استخراج CLS token
        cls_output = embeddings[:, 0, :] # بعد: [Rank]
        
        # ✅ اینجا ورودی classifier برابر rank است و وزن‌ها هم rank هستند -> OK
        logits = self.classifier(cls_output)
        
        return {"logits": logits}
        
class BaseLayerCompressed(nn.Module):
    """
    لایه انکودر کاملاً فشرده شده.
    - ورودی: Hidden State با بعد برابر 'rank' (نه hidden_size).
    - خروجی: Hidden State با بعد برابر 'rank'.
    - هیچ پروجکشنی از 768 به Rank وجود ندارد، چون ورودی خودش Rank است.
    """
    def __init__(self, rank: int, config):
        super().__init__()
        self.rank = rank
        
        # لایه توجه بدون Q و K (فقط Value)
        # ورودی: [B, L, Rank] -> خروجی: [B, L, Rank]
        self.attention_value = nn.Linear(rank, rank, bias=True)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attn_norm = nn.LayerNorm(rank)
        
        # لایه خروجی (Output Dense)
        # در معماری BERT، لایه خروجی ورودی را به hidden_size برمی‌گرداند.
        # اما اینجا چون کل مدل در فضای Rank است، ما باید خروجی را هم در همان Rank نگه داریم
        # تا بتوانیم با ورودی اصلی (که Rank است) جمع بزنیم.
        # پس این لایه باید: Input=Rank -> Output=Rank
        self.output_dense = nn.Linear(rank, rank, bias=True)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(rank)
        
        # FFN استاندارد (با ابعاد فشرده)
        # ورودی: Rank -> Hidden_Size (مثلا 4*Rank) -> خروجی: Rank
        intermediate_size = max(4 * rank, 32) # حداقل سایز 32 برای ثبات
        self.intermediate = nn.Linear(rank, intermediate_size, bias=True)
        self.output_ffn = nn.Linear(intermediate_size, rank, bias=True)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        
        # 1. اعمال Attention (ورودی و خروجی هر دو Rank هستند)
        attn_out = self.attention_value(hidden_states) # [B, L, Rank]
        attn_out = self.attn_norm(attn_out)
        attn_out = self.attn_dropout(attn_out)
        
        # جمع زدن با رزیدوال (هر دو Rank هستند)
        hidden_states = hidden_states + attn_out
        hidden_states = self.output_norm(hidden_states)
        
        # 2. FFN
        intermediate = self.intermediate(hidden_states)
        intermediate = torch.relu(intermediate)
        ff_out = self.output_ffn(intermediate)
        ff_out = self.ffn_dropout(ff_out)
        
        # جمع زدن با رزیدوال
        hidden_states = hidden_states + ff_out
        
        return (hidden_states,)
    # ==================== TRAINING & EVALUATION ====================
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, config: ExperimentConfig, num_labels: int, is_multichoice: bool = False):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_loss = 0.0
    steps_per_epoch = len(train_loader)
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                optimizer.zero_grad(set_to_none=True)
                loss = None
                
                if is_multichoice:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    true_labels = batch['labels'].to(device)
                    
                    batch_size, num_choices, seq_len = input_ids.shape
                    input_ids_flat = input_ids.view(-1, seq_len)
                    attention_mask_flat = attention_mask.view(-1, seq_len)
                    
                    outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                    logits = outputs['logits']
                    
                    combined_labels = true_labels.repeat_interleave(num_choices)
                    if combined_labels.size(0) != logits.size(0):
                         raise RuntimeError(f"Label/Logit mismatch: {combined_labels.size()} vs {logits.size()}")

                    loss = nn.functional.cross_entropy(logits, combined_labels)
                    
                else:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    loss = nn.functional.cross_entropy(logits, labels)
                
                if loss is not None and torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                else:
                    continue

            except RuntimeError as e:
                error_msg = str(e)
                if "backward through the graph" in error_msg or "detected an anomaly" in error_msg:
                    print(f"\n⛔ Graph Error in batch {batch_idx}. Clearing cache...")
                    if device.type == 'mps': torch.mps.empty_cache()
                    elif device.type == 'cuda': torch.cuda.empty_cache()
                    gc.collect()
                    model.train() 
                continue
            except Exception as e:
                print(f"\n❌ Unexpected Error in batch {batch_idx}: {e}")
                continue
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}' if loss is not None else 'N/A'})
        
        avg_epoch_loss = epoch_loss / max(1, steps_per_epoch)
        print(f"   ✅ Epoch {epoch+1} Completed. Avg Loss: {avg_epoch_loss:.4f}")

    val_accuracy = evaluate_model(model, val_loader, device, num_labels, is_multichoice=is_multichoice)
    return {'final_accuracy': val_accuracy, 'avg_loss': avg_epoch_loss}

def evaluate_model(model, data_loader, device, num_labels, is_multichoice: bool = False):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                if is_multichoice:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    true_labels = batch['labels']
                    
                    batch_size, num_choices, _ = input_ids.shape
                    input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
                    attention_mask_flat = attention_mask.view(-1, input_ids.shape[-1])
                    
                    outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                    logits = outputs['logits']
                    
                    logits_reshaped = logits.view(batch_size, num_choices, -1)
                    avg_logits = logits_reshaped.mean(dim=1)
                    preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(true_labels.numpy())
                
                else:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(batch['labels'].numpy())
            
            except Exception as e:
                continue
    
    if len(all_preds) == 0 or len(all_labels) == 0:
        return 0.0
    return accuracy_score(all_labels, all_preds)

def benchmark_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    start_time = time.time()
    batch_times = []
    
    with torch.no_grad():
        for batch in data_loader:
            b_start = time.time()
            if isinstance(batch['input_ids'], list) or batch['input_ids'].dim() == 3:
                if batch['input_ids'].dim() == 3:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    for i in range(input_ids.shape[1]):
                        _ = model(input_ids=input_ids[:, i, :], attention_mask=attention_mask[:, i, :])
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            batch_times.append(time.time() - b_start)
    
    total_time = time.time() - start_time
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    return {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'samples_per_second': len(data_loader.dataset) / total_time if total_time > 0 else 0,
        'memory_usage_mb': memory_after - memory_before,
        'peak_memory_mb': memory_after,
        'throughput': len(data_loader.dataset) / total_time if total_time > 0 else 0
    }

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==================== MAIN EXPERIMENT PIPELINE ====================
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, experiment_id: int, results: List[Dict], checkpoint_name: str = "checkpoint.json"):
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_data = {'experiment_id': experiment_id, 'results': results, 'timestamp': time.time()}
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.json") -> Tuple[int, List[Dict]]:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return 0, []
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data['experiment_id'], checkpoint_data['results']
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return 0, []

def create_comprehensive_plots(results_df: pd.DataFrame, plots_dir: str):
    print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS...")
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('SVD Compression Analysis: Comprehensive Results', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax1.plot(model_data['rank'], model_data['compressed_accuracy'], marker='o', label=model, linewidth=2)
    ax1.set_xlabel('SVD Rank'); ax1.set_ylabel('Accuracy'); ax1.set_title('Accuracy vs Rank (by Model)'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        ax2.plot(dataset_data['rank'], dataset_data['compressed_accuracy'], marker='s', label=dataset, linewidth=2)
    ax2.set_xlabel('SVD Rank'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy vs Rank (by Dataset)'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        speedup_by_rank = model_data.groupby('rank')['speedup_factor'].mean()
        ax3.plot(speedup_by_rank.index, speedup_by_rank.values, marker='^', label=model, linewidth=2)
    ax3.set_xlabel('SVD Rank'); ax3.set_ylabel('Speedup Factor'); ax3.set_title('Inference Speedup vs Rank'); ax3.legend(); ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 0]
    param_data = results_df.groupby(['model', 'rank'])['param_reduction'].mean().reset_index()
    for model in param_data['model'].unique():
        model_param_data = param_data[param_data['model'] == model]
        ax4.bar(model_param_data['rank'] + (ord(model[0]) - 97) * 0.2, model_param_data['param_reduction'], width=0.2, label=model)
    ax4.set_xlabel('SVD Rank'); ax4.set_ylabel('Parameters Reduced'); ax4.set_title('Parameter Reduction by Model'); ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = axes[1, 1]
    accuracy_drops = results_df['accuracy_drop']
    ax5.hist(accuracy_drops, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel('Accuracy Drop'); ax5.set_ylabel('Frequency'); ax5.set_title('Distribution of Accuracy Drops'); ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    memory_data = results_df.groupby('rank')['memory_reduction'].mean().reset_index()
    ax6.bar(memory_data['rank'], memory_data['memory_reduction'], color='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax6.set_xlabel('SVD Rank'); ax6.set_ylabel('Memory Reduction (MB)'); ax6.set_title('Average Memory Reduction'); ax6.grid(True, alpha=0.3, axis='y')
    
    ax7 = axes[2, 0]
    task_data = results_df.groupby(['task_type', 'rank'])['compressed_accuracy'].mean().reset_index()
    for task_type in task_data['task_type'].unique():
        task_type_data = task_data[task_data['task_type'] == task_type]
        ax7.plot(task_type_data['rank'], task_type_data['compressed_accuracy'], marker='D', label=task_type, linewidth=2)
    ax7.set_xlabel('SVD Rank'); ax7.set_ylabel('Accuracy'); ax7.set_title('Accuracy by Task Type'); ax7.legend(); ax7.grid(True, alpha=0.3)
    
    ax8 = axes[2, 1]
    pivot_data = results_df.pivot_table(index='model', columns='rank', values='compression_ratio', aggfunc='mean')
    im = ax8.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    ax8.set_xticks(range(len(pivot_data.columns))); ax8.set_xticklabels(pivot_data.columns)
    ax8.set_yticks(range(len(pivot_data.index))); ax8.set_yticklabels(pivot_data.index)
    ax8.set_xlabel('SVD Rank'); ax8.set_ylabel('Model'); ax8.set_title('Compression Ratio Heatmap')
    plt.colorbar(im, ax=ax8, label='Compression Ratio')
    
    ax9 = axes[2, 2]
    best_configs = []
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        best_idx = dataset_data['compressed_accuracy'].idxmax()
        best_configs.append(dataset_data.loc[best_idx])
    
    if best_configs:
        best_df = pd.DataFrame(best_configs)
        y_pos = np.arange(len(best_df))
        ax9.barh(y_pos, best_df['compressed_accuracy'], color='skyblue', edgecolor='navy')
        ax9.set_yticks(y_pos); ax9.set_yticklabels(best_df['dataset'])
        ax9.set_xlabel('Accuracy'); ax9.set_title('Best Configuration for Each Dataset')
        for i, (acc, model, rank) in enumerate(zip(best_df['compressed_accuracy'], best_df['model'], best_df['rank'])):
            ax9.text(acc + 0.01, i, f'{model} (R{rank})', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(plots_dir) / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(config: ExperimentConfig, device: torch.device, logger: logging.Logger):
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.plots_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_manager = CheckpointManager(config.output_dir)
    
    logger.info("🔍 Discovering datasets...")
    config.datasets = discover_datasets(config)
    
    if config.selected_datasets:
        config.datasets = [d for d in config.datasets if d.name in config.selected_datasets]
    
    if not config.datasets:
        logger.error("❌ No valid datasets found!")
        return pd.DataFrame()
    
    logger.info(f"🎯 Selected datasets: {[d.name for d in config.datasets]}")
    
    experiment_id, all_results = checkpoint_manager.load_checkpoint()
    logger.info(f"🔄 Starting from experiment ID: {experiment_id}")
    
    models_to_test = list(config.base_models.items())
    if config.selected_models:
        models_to_test = [(k, v) for k, v in models_to_test if k in config.selected_models]
    
    ranks_to_test = config.svd_ranks
    if config.selected_ranks:
        ranks_to_test = [r for r in ranks_to_test if r in config.selected_ranks]
    
    for model_name, model_path in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 TESTING MODEL: {model_name.upper()} ({model_path})")
        logger.info(f"{'='*60}")
        
        for dataset_config in config.datasets:
            logger.info(f"\n   📝 DATASET: {dataset_config.name} (task_type={dataset_config.task_type}, num_labels={dataset_config.num_labels})")
            
            try:
                df = pd.read_csv(dataset_config.path)
                if len(df) > config.max_samples:
                    df = df.sample(n=config.max_samples, random_state=config.seed)
                
                split_idx = int(len(df) * (1 - config.val_split))
                train_df = df.iloc[:split_idx].reset_index(drop=True)
                val_df = df.iloc[split_idx:].reset_index(drop=True)
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                is_multichoice = (dataset_config.task_type == 'mc')
                
                if is_multichoice:
                    train_dataset = MultiChoiceDataset(train_df, tokenizer, config.max_length)
                    val_dataset = MultiChoiceDataset(val_df, tokenizer, config.max_length)
                    
                    if len(train_dataset) == 0 or len(val_dataset) == 0:
                        logger.error(f"❌ No valid samples for {dataset_config.name} after filtering.")
                        continue
                else:
                    train_dataset = SingleLabelDataset(train_df, tokenizer, config.max_length)
                    val_dataset = SingleLabelDataset(val_df, tokenizer, config.max_length)
                
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
                
                # --- ساخت مدل پایه برای مقایسه (اختیاری، اگر بخواهید دقت پایه را ببینید) ---
                # در اینجا ما مستقیماً روی مدل فشرده تمرکز می‌کنیم تا خطا ندهد.
                # اگر نیاز به مدل پایه دارید، می‌توانید از AutoModelForSequenceClassification استفاده کنید.
                
                logger.info(f"   🏋️ Training Compressed Model (Rank={config.svd_ranks[0]} as baseline)...")
                # محاسبه پارامترهای اولیه مدل فشرده (با اولین رنک)
                # نکته: برای سادگی، ما فقط مدل فشرده را آموزش می‌دهیم.
                
                base_model_stats = None
                base_accuracy = 0.0
                
                for rank in ranks_to_test:
                    experiment_id += 1
                    logger.info(f"\n      🔬 Rank {rank}:")
                    
                    try:
                        # ساخت مدل فشرده شده با معماری جدید
                        logger.info(f"         🗜️ Initializing Compressed BERT (Rank={rank})...")
                        model = CompressedBertModel(
                            model_path=model_path, 
                            rank=rank, 
                            num_labels=dataset_config.num_labels
                        ).to(device)
                        
                        # شمارش پارامترها
                        params_after = count_parameters(model)
                        logger.info(f"         ✅ Model Initialized | Params: {params_after:,}")
                        
                        # بچمارکینگ سرعت (قبل از آموزش)
                        logger.info("         📈 Benchmarking compressed model...")
                        compressed_stats = benchmark_model(model, val_loader, device)
                        
                        # آموزش
                        logger.info("         ⚡ Training compressed model...")
                        compressed_results = train_model(
                            model, train_loader, val_loader, device, 
                            config, dataset_config.num_labels, is_multichoice=is_multichoice
                        )
                        
                        # محاسبه سرعت نسبی (فرض بر اینکه مدل پایه با همان ابعاد اجرا شده یا مقایسه کلی)
                        # چون مدل پایه (768) با مدل فشرده (8) متفاوت است، سرعت مستقیم قابل مقایسه نیست مگر با تنظیمات خاص.
                        # اما ما speedup را نسبت به یک مدل فرضی 768 محاسبه می‌کنیم یا فقط گزارش می‌دهیم.
                        # برای دقت بیشتر، ما فقط زمان اجرای خود مدل را گزارش می‌دهیم.
                        
                        result = {
                            'experiment_id': experiment_id,
                            'model': model_name,
                            'dataset': dataset_config.name,
                            'task_type': dataset_config.task_type,
                            'num_labels': dataset_config.num_labels,
                            'rank': rank,
                            
                            'base_accuracy': base_accuracy, # در اینجا 0 است مگر اینکه مدل پایه هم ساخته شود
                            'compressed_accuracy': compressed_results['final_accuracy'],
                            'accuracy_drop': base_accuracy - compressed_results['final_accuracy'],
                            
                            'base_params': 0, # یا مقدار واقعی مدل پایه اگر محاسبه شود
                            'compressed_params': params_after,
                            'param_reduction': 0, # یا تفاوت با مدل پایه
                            'compression_ratio': 0.0, # محاسبه نسبت پارامترها اگر مدل پایه موجود باشد
                            
                            'base_batch_time': 0.0,
                            'compressed_batch_time': compressed_stats['avg_batch_time'],
                            'speedup_factor': 1.0, # چون مدل پایه نداریم، فرض می‌کنیم 1 است
                            
                            'base_memory_mb': 0.0,
                            'compressed_memory_mb': compressed_stats['memory_usage_mb'],
                            'memory_reduction': 0.0,
                            
                            'compressed_layers': 0 # تعداد لایه‌های فشرده شده
                        }
                        
                        all_results.append(result)
                        
                        logger.info(f"         ✅ Complete | Acc: {compressed_results['final_accuracy']:.4f} | "
                                   f"Params: {params_after:,} | "
                                   f"Time/Batch: {compressed_stats['avg_batch_time']:.4f}s")
                        
                        if experiment_id % config.save_checkpoint_interval == 0:
                            checkpoint_manager.save_checkpoint(experiment_id, all_results)
                        
                        del model
                        gc.collect()
                        if device.type == 'mps': torch.mps.empty_cache()
                        elif device.type == 'cuda': torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.error(f"         ❌ Error in rank {rank}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                del tokenizer
                gc.collect()
                
            except Exception as e:
                logger.error(f"   ❌ Error processing dataset {dataset_config.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    results_df = pd.DataFrame(all_results)
    results_path = Path(config.output_dir) / "compression_results.csv"
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"\n✅ All experiments completed! Results saved to {results_path}")
    
    if not results_df.empty:
        create_comprehensive_plots(results_df, config.plots_dir)
    
    return results_df


# ==================== CLI INTERFACE ====================
def parse_arguments():
    """پارسی کردن آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(description='SVD Compression Experiments for Reasoning Tasks')
    
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['bert', 'distilbert', 'roberta'],
                       help='مدل‌هایی که باید تست شوند (پیش‌فرض: همه)')
    
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='دیتاست‌هایی که باید تست شوند (پیش‌فرض: همه موجود)')
    
    parser.add_argument('--ranks', type=int, nargs='+',
                       help='رنک‌های SVD مورد آزمایش (پیش‌فرض: [8, 16, 32, 64])')
    
    parser.add_argument('--max-samples', type=int, default=200,
                       help='حداکثر نمونه برای هر دیتاست (پیش‌فرض: 200)')
    
    parser.add_argument('--batch-size', type=int, default=4,
                       help='سایز بچ (پیش‌فرض: 4)')
    
    parser.add_argument('--epochs', type=int, default=1,
                       help='تعداد اپوک آموزش (پیش‌فرض: 1)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed تصادفی (پیش‌فرض: 42)')
    
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='مسیر خروجی (پیش‌فرض: ./results)')
    
    parser.add_argument('--resume', action='store_true',
                       help='ادامه آزمایش از آخرین چک‌پوینت')
    
    return parser.parse_args()


# ==================== MAIN EXECUTION ====================
def main():
    """تابع اجرایی اصلی"""
    args = parse_arguments()
    
    # ایجاد کانفیگ بر اساس ورودی‌های کاربر
    config = ExperimentConfig(
        selected_models=args.models if args.models else [],
        selected_datasets=args.datasets if args.datasets else [],
        selected_ranks=args.ranks if args.ranks else [],
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # تنظیم محیط آزمایش
    device, logger = setup_experiment(config)
    
    # اجرای آزمایش‌ها
    try:
        results_df = run_experiment(config, device, logger)
        
        # چاپ خلاصه آمار
        if not results_df.empty:
            logger.info("\n" + "="*80)
            logger.info("📊 EXPERIMENT SUMMARY")
            logger.info("="*80)
            
            # آمار کلی
            avg_accuracy_drop = results_df['accuracy_drop'].mean()
            avg_speedup = results_df['speedup_factor'].mean()
            avg_compression = results_df['compression_ratio'].mean()
            
            logger.info(f"Average Accuracy Drop: {avg_accuracy_drop:.4f}")
            logger.info(f"Average Speedup: {avg_speedup:.2f}x")
            logger.info(f"Average Compression Ratio: {avg_compression:.2%}")
            
            # بهترین پیکربندی‌ها
            best_overall = results_df.loc[results_df['compressed_accuracy'].idxmax()]
            best_speedup = results_df.loc[results_df['speedup_factor'].idxmax()]
            
            logger.info(f"\n🏆 Best Overall Accuracy:")
            logger.info(f"   Model: {best_overall['model']}")
            logger.info(f"   Dataset: {best_overall['dataset']}")
            logger.info(f"   Rank: {best_overall['rank']}")
            logger.info(f"   Accuracy: {best_overall['compressed_accuracy']:.4f}")
            
            logger.info(f"\n⚡ Best Speedup:")
            logger.info(f"   Model: {best_speedup['model']}")
            logger.info(f"   Dataset: {best_speedup['dataset']}")
            logger.info(f"   Rank: {best_speedup['rank']}")
            logger.info(f"   Speedup: {best_speedup['speedup_factor']:.2f}x")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # پاکسازی نهایی
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
