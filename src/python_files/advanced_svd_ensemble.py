
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import gc
import warnings
import traceback
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import optuna
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    # مسیرها
    data_dir: str = "./data"
    output_dir: str = "./advanced_consolidation_results"
    cache_dir: Optional[str] = None
    
    base_models: List[str] = None 
    
    # پارامترهای SVD
    min_rank: int = 8
    max_rank: int = 128
    energy_threshold: float = 0.9
    
    # آموزش
    batch_size: int = 8
    max_length: int = 128
    num_labels: int = 2
    
    # Ensemble
    ensemble_method: str = "hybrid"
    temperature: float = 2.0
    
    enable_hyperparam_opt: bool = False  
    n_trials: int = 20
    
    def __post_init__(self):
        if self.base_models is None:
            self.base_models = [
                "distilbert-base-uncased-finetuned-sst-2-english",
                "roberta-base", 
                "albert-base-v2"
            ]

config = Config()




# ==================== 4. FINE-TUNING HELPER FUNCTION ====================
import torch
import torch.nn.functional as F
import gc

def fine_tune_compressed_model(model, train_dataloader, val_dataloader, device, epochs=3, lr=2e-5):
    """
    آموزش مجدد (Fine-tuning) مدل فشرده شده روی داده‌های اختصاصی تسک
    نسخه کامل با ارزیابی و مدیریت حافظه
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"   🔥 Starting Fine-tuning on {device}... Total batches: {len(train_dataloader)}", flush=True)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # --- مرحله آموزش (Training Phase) ---
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # انتقال داده‌ها به دستگاه (CPU/GPU/MPS)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                # اجرای مدل
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = F.cross_entropy(logits, labels)
                
                # محاسبه گرادیان و آپدیت وزن‌ها
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # محاسبه دقت موقت
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # نمایش پیشرفت هر ۲ بچ برای اطمینان از حرکت پردازشگر
                if batch_idx % 2 == 0:
                    print(f"      [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dataloader)} | Loss: {loss.item():.4f}", flush=True)
                    
            except Exception as e:
                print(f"\n❌ ERROR at batch {batch_idx}: {e}", flush=True)
                raise e
        
        # محاسبه میانگین خطا و دقت آموزش
        avg_train_loss = total_loss / len(train_dataloader)
        train_acc = correct / total
        
        # --- مرحله ارزیابی (Validation Phase) ---
        print(f"\n   📊 Evaluating Validation Set...", flush=True)
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        # نمایش نتایج پایان اپوک
        print(f"   ✅ Epoch {epoch+1}/{epochs} Finished -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}", flush=True)
        
        # ذخیره بهترین مدل
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   🏆 New Best Model Found! Accuracy: {best_val_acc:.4f}", flush=True)
        
        # بازگشت به حالت آموزش برای اپوک بعد
        model.train()
        
        # پاکسازی حافظه اضافی در پایان هر اپوک
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return model, best_val_acc



# ==================== 1. ADAPTIVE SVD COMPRESSION ====================
class AdaptiveSVDLinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, energy_threshold: float = 0.9, device: str = "mps"):
        super().__init__()
        
        W = original_layer.weight.data.cpu().numpy()
        
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        
        energy = np.cumsum(S) / np.sum(S)
        rank = np.argmax(energy >= energy_threshold) + 1
        rank = max(config.min_rank, min(rank, min(W.shape)))
        
        self.rank = rank
        
        # انتقال صریح به دستگاه مورد نظر
        self.U = nn.Parameter(torch.from_numpy(U[:, :rank]).float().to(device))
        self.S = nn.Parameter(torch.from_numpy(S[:rank]).float().to(device))
        self.V = nn.Parameter(torch.from_numpy(Vh[:rank, :]).float().to(device))
        
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone().to(device))
        else:
            self.bias = None
        
        original_params = W.size
        compressed_params = U[:, :rank].size + S[:rank].size + Vh[:rank, :].size
        self.compression_ratio = compressed_params / original_params
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_recon = self.U @ torch.diag(self.S) @ self.V
        return F.linear(x, W_recon, self.bias)
    
    def get_importance_score(self) -> float:
        return torch.sum(self.S).item()

class AdaptiveSVDCompressor:    
    def __init__(self, energy_threshold: float = 0.9, device: str = "mps"):
        self.energy_threshold = energy_threshold
        self.device = device
        self.layer_stats = []
    
    def compress_model(self, model: PreTrainedModel, model_name: str) -> Tuple[PreTrainedModel, Dict]:
        stats = {
            'original_params': sum(p.numel() for p in model.parameters()),
            'compressed_params': 0,
            'layers_compressed': 0,
            'avg_compression_ratio': 0,
            'importance_scores': []
        }
        
        skip_keywords = ['classifier', 'embeddings', 'pooler', 'lm_head']
        
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                if any(kw in name.lower() for kw in skip_keywords):
                    continue
                
                try:
                    compressed_layer = AdaptiveSVDLinear(module, self.energy_threshold, device=self.device)
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    setattr(parent, child_name, compressed_layer)
                    
                    stats['layers_compressed'] += 1
                    stats['importance_scores'].append({
                        'layer': name,
                        'importance': compressed_layer.get_importance_score(),
                        'compression_ratio': compressed_layer.compression_ratio
                    })
                    
                except Exception as e:
                    print(f"⚠️ Failed to compress layer {name}: {e}")
        
        stats['compressed_params'] = sum(p.numel() for p in model.parameters())
        if stats['layers_compressed'] > 0:
            stats['avg_compression_ratio'] = np.mean([
                s['compression_ratio'] for s in stats['importance_scores']
            ])
        
        print(f"✅ Compressed {model_name}: {stats['layers_compressed']} layers, "
              f"Ratio: {stats['compressed_params']/stats['original_params']:.2%}")
        
        return model, stats


# ==================== 2. TASK-AWARE ATTENTION ====================
class TaskAwareAttention(nn.Module):
    def __init__(self, num_tasks: int, hidden_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        
        # امبدینگ برای هر تسک (نه هر مدل)
        self.task_embeddings = nn.Embedding(num_tasks, hidden_dim)
        
        # لایه‌های تبدیل
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ماتریس شباهت بین تسک‌ها
        self.similarity_matrix = nn.Parameter(torch.eye(num_tasks))
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        nn.init.xavier_uniform_(self.task_embeddings.weight)
        nn.init.xavier_uniform_(self.similarity_matrix)
        
    def forward(self, task_id: int, all_logits: torch.Tensor, model_task_ids: List[int]) -> torch.Tensor:
        """
        ورودی‌ها:
            task_id: شناسه تسک فعلی که داریم روی آن تست می‌زنیم
            all_logits: تانسور شکل (num_models, batch_size, num_labels)
            model_task_ids: لیستی از شناسه تسک‌های مربوط به هر مدل (مثلاً [0,0,0, 1,1,1, ...])
        """
        device = all_logits.device
        batch_size = all_logits.shape[1]
        num_models = all_logits.shape[0]
        
        # 1. محاسبه امبدینگ تسک هدف (task_id)
        current_task_emb = self.task_embeddings(torch.tensor([task_id], device=device))
        current_task_emb = self.query_proj(current_task_emb)
        
        # 2. محاسبه امبدینگ تمام تسک‌های موجود در لیست model_task_ids
        # تبدیل لیست به تانسر
        target_task_ids_tensor = torch.tensor(model_task_ids, device=device)
        all_target_task_embs = self.task_embeddings(target_task_ids_tensor)
        all_target_task_embs = self.key_proj(all_target_task_embs)
        
        # 3. محاسبه امتیاز توجه (Query-Key)
        # current_task_emb: (1, hidden_dim)
        # all_target_task_embs: (num_models, hidden_dim)
        attention_scores = torch.matmul(current_task_emb, all_target_task_embs.T) / np.sqrt(self.hidden_dim)
        
        # 4. اضافه کردن شباهت یاد گرفته شده بین تسک‌ها
        # similarity_matrix: (num_tasks, num_tasks)
        # ما نیاز داریم مقادیر مربوط به تسک‌های موجود در model_task_ids را برداریم
        task_similarities = self.similarity_matrix[target_task_ids_tensor, task_id].unsqueeze(0)
        
        combined_scores = attention_scores + task_similarities
        
        # 5. نرمال‌سازی (Softmax) روی بعد مدل‌ها
        weights = F.softmax(combined_scores, dim=-1)  # شکل: (1, num_models)
        
        # 6. گسترش ابعاد برای ضرب با logits
        # weights: (1, num_models) -> (1, num_models, 1, 1)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        
        # 7. ترکیب وزنی
        weighted_logits = (all_logits * weights_expanded).sum(dim=1)
        
        return weighted_logits.squeeze(0)
    
    def update_similarity_matrix(self, performance_matrix: torch.Tensor):
        normalized_perf = F.softmax(performance_matrix, dim=-1)
        self.similarity_matrix.data = 0.7 * self.similarity_matrix.data + 0.3 * normalized_perf
        
        
# ==================== 3. SMART WEIGHTED ENSEMBLE ====================
class SmartWeightedEnsemble:
    def __init__(self, num_models: int, num_tasks: int):
        self.num_models = num_models
        self.num_tasks = num_tasks
        
        self.performance_matrix = torch.ones(num_models, num_tasks) * 0.5
        
        self.base_weights = torch.ones(num_models) / num_models
        
        self.attention = TaskAwareAttention(num_tasks)
        
        self.confidence_cache = {}
        
    def compute_model_confidence(self, model, input_ids, attention_mask, temperature: float = 2.0) -> float:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            probs = F.softmax(logits / temperature, dim=-1)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            confidence = 1.0 / (1.0 + entropy.mean().item())
            
            return confidence
    
    def compute_task_similarity(self, model_id: int, task_id: int) -> float:
        model_perf_on_task = self.performance_matrix[model_id, task_id].item()
        avg_perf = self.performance_matrix[:, task_id].mean().item()
        
        if avg_perf > 0:
            similarity = model_perf_on_task / avg_perf
        else:
            similarity = 1.0
        
        return min(2.0, max(0.5, similarity))
    
    def get_dynamic_weights(self, task_id: int, models, input_ids, attention_mask) -> torch.Tensor:
        weights = torch.zeros(self.num_models).to(input_ids.device)
        
        for i, model in enumerate(models):
            w_base = self.base_weights[i].item()
            w_performance = self.performance_matrix[i, task_id].item()
            
            cache_key = f"{i}_{task_id}_{input_ids.sum().item()}"
            if cache_key not in self.confidence_cache:
                confidence = self.compute_model_confidence(model, input_ids, attention_mask)
                self.confidence_cache[cache_key] = confidence
            w_confidence = self.confidence_cache[cache_key]
            
            w_similarity = self.compute_task_similarity(i, task_id)
            
            final_weight = (w_base * w_performance * w_confidence * w_similarity) ** 0.25
            weights[i] = final_weight
        
        weights = weights / (weights.sum() + 1e-10)
        
        return weights
    
    def update_performance_matrix(self, model_id: int, task_id: int, accuracy: float):
        old_value = self.performance_matrix[model_id, task_id].item()
        new_value = 0.8 * old_value + 0.2 * accuracy
        self.performance_matrix[model_id, task_id] = new_value
        self.attention.update_similarity_matrix(self.performance_matrix)
    
    def combine_logits(self, task_id: int, models, all_logits: torch.Tensor, 
                      input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      method: str = "hybrid") -> torch.Tensor:
        
        if method == "simple_average":
            return all_logits.mean(dim=0)
        
        elif method == "weighted":
            weights = self.get_dynamic_weights(task_id, models, input_ids, attention_mask)
            weights = weights.view(-1, 1, 1).to(all_logits.device)
            return (all_logits * weights).sum(dim=0)
        
        elif method == "task_aware_attention":
            # ⭐ نکته کلیدی: ساخت لیست تسک‌های مربوط به هر مدل
            # فرض کنید هر مدل در model_info دارای 'task_id' است
            # اگر در اینجا دسترسی ندارید، باید از اطلاعاتی که در create_smart_ensemble ذخیره کردید استفاده کنید
            
            # روش ساده: فرض می‌کنیم مدل‌ها به ترتیب تسک‌ها مرتب شده‌اند یا باید از model_info استفاده کنیم
            # اگر model_info را دارید:
            # model_task_ids = [info['task_id'] for info in self.model_info]
            
            # اگر model_info را ندارید، باید یک لیست دستی بسازید (مثلاً 3 مدل برای هر تسک)
            # فرض کنید 3 مدل پایه دارید و برای هر تسک 3 مدل ساخته شده:
            num_base_models = len(self.config.base_models) # مثلا 3
            # ساخت لیست تسک‌ها: [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4]
            model_task_ids = []
            for i in range(len(models)):
                # این قسمت بستگی به نحوه ساخت مدل‌ها دارد. 
                # اگر مدل‌ها به صورت گروهی ساخته شده‌اند، می‌توان از index استفاده کرد:
                model_task_ids.append(i // num_base_models)
            
            return self.attention(task_id, all_logits, model_task_ids)
        
        elif method == "hybrid":
            # مشابه بالا برای hybrid
            num_base_models = len(self.config.base_models)
            model_task_ids = [i // num_base_models for i in range(len(models))]
            
            dynamic_weights = self.get_dynamic_weights(task_id, models, input_ids, attention_mask)
            dynamic_weights = dynamic_weights.view(-1, 1, 1).to(all_logits.device)
            
            attention_weighted = self.attention(task_id, all_logits, model_task_ids)
            dynamic_weighted = (all_logits * dynamic_weights).sum(dim=0)
            
            combined = 0.6 * attention_weighted + 0.4 * dynamic_weighted
            
            return combined
        
        else:
            raise ValueError(f"Unknown method: {method}")


# ==================== 4. KNOWLEDGE DISTILLATION ====================
class KnowledgeDistiller:    
    def __init__(self, teacher_ensemble, temperature: float = 2.0, alpha: float = 0.7):
        self.teacher = teacher_ensemble
        self.temperature = temperature
        self.alpha = alpha
        
    def distill(self, student_model, dataloaders, num_epochs: int = 3, lr: float = 1e-4):
        student_model.train()
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)
        
        all_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for task_id, dataloader in enumerate(dataloaders):
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(next(student_model.parameters()).device)
                    attention_mask = batch['attention_mask'].to(input_ids.device)
                    labels = batch['labels'].to(input_ids.device)
                    
                    with torch.no_grad():
                        teacher_logits = self.teacher.combine_logits(
                            task_id, self.teacher.models, 
                            self.teacher.get_all_logits(input_ids, attention_mask),
                            input_ids, attention_mask
                        )
                        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                    
                    student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
                    
                    student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
                    distillation_loss = F.kl_div(
                        student_probs, teacher_probs, 
                        reduction='batchmean'
                    ) * (self.temperature ** 2)
                    
                    hard_loss = F.cross_entropy(student_logits, labels)
                    
                    total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            all_losses.append(avg_loss)
            print(f"Distillation Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return student_model, all_losses

# ==================== 5. DATA HANDLING ====================
class MultiTaskDataset(Dataset):    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_all_tasks(data_dir: str):
    csv_files = sorted(glob.glob(f"{data_dir}/*.csv"))
    tasks = []
    
    print(f"📂 Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            text_cols = [c for c in df.columns if 'text' in c.lower()]
            label_cols = [c for c in df.columns if 'label' in c.lower()]
            
            if not text_cols or not label_cols:
                print(f"⚠️ Skipping {Path(file_path).name}: No text/label columns")
                continue
            
            text_col = text_cols[0]
            label_col = label_cols[0]
            
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df = df.dropna(subset=['text', 'label'])
            
            # محدود کردن به 500 نمونه
            df = df.head(500) 

            if len(df) < 5:
                print(f"⚠️ Skipping {Path(file_path).name}: Too few samples ({len(df)})")
                continue
            
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            val_df = df.iloc[split_idx:].reset_index(drop=True)
            
            tasks.append({
                'name': Path(file_path).stem,
                'train_df': train_df,
                'val_df': val_df,
                'file_path': file_path,
                'total_samples': len(df)
            })
            
            print(f"   ✅ {Path(file_path).name}: {len(df)} samples loaded (Max 500 applied)")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print(f"🎯 Total tasks loaded: {len(tasks)}")

    # محدود کردن به 5 تسک اول
    if len(tasks) > 5:
        print(f"⚠️ Limiting execution to the first 5 tasks only...")
        tasks = tasks[:5]

    return tasks  # ⭐ اصلاح مهم: برگرداندن لیست tasks

# ==================== 6. MAIN ADVANCED ENSEMBLE SYSTEM ====================
class AdvancedSVDAssembler:    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.compressor = AdaptiveSVDCompressor(energy_threshold=config.energy_threshold, device=str(self.device))
        self.ensemble = None
        self.distiller = None
        self.consolidator = None
        
        self.results = {
            'tasks': [],
            'models': [],
            'performance_matrix': None,
            'ensemble_performance': {},
            'compression_stats': []
        }
        
        Path(config.output_dir).mkdir(exist_ok=True, parents=True)
        
        print(f"🚀 Advanced SVD Assembler Initialized on {self.device}")
    
    def compress_all_models(self, tasks):
        compressed_models = []
        
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"🔧 Compressing models for Task {task_idx+1}/{len(tasks)}: {task['name']}")
            print(f"{'='*60}")
            
            task_models = []
            task_stats = []
            
            for model_idx, model_name in enumerate(self.config.base_models):
                try:
                    print(f"   Loading {model_name}...")
                    
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=self.config.num_labels,
                        cache_dir=self.config.cache_dir,
                        ignore_mismatched_sizes=True
                    )
                    
                    # ⭐ حیاتی: انتقال مدل به دستگاه قبل از فشرده‌سازی
                    model = model.to(self.device)
                    
                    compressed_model, stats = self.compressor.compress_model(model, model_name)
                    
                    stats['model_name'] = model_name
                    stats['task_name'] = task['name']
                    task_stats.append(stats)
                    
                    task_models.append({
                        'model': compressed_model,
                        'name': f"{model_name}_{task['name']}",
                        'base_model': model_name,
                        'task_id': task_idx,
                        'stats': stats
                    })
                    
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    
                except Exception as e:
                    print(f"   ❌ Error with {model_name}: {e}")
                    continue
            
            if task_models:
                compressed_models.append({
                    'task': task,
                    'models': task_models,
                    'task_id': task_idx,
                    'stats': task_stats
                })
                
                self.results['compression_stats'].extend(task_stats)
        
        return compressed_models
    
    def evaluate_cross_task(self, compressed_models, tasks):
        print(f"\n{'='*60}")
        print("📊 CROSS-TASK EVALUATION")
        print(f"{'='*60}")
        
        num_models = sum(len(cm['models']) for cm in compressed_models)
        num_tasks = len(tasks)
        
        performance_matrix = torch.zeros(num_models, num_tasks)
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_models[0])
        val_dataloaders = []
        
        for task in tasks:
            val_ds = MultiTaskDataset(
                task['val_df']['text'],
                task['val_df']['label'],
                tokenizer,
                max_length=self.config.max_length
            )
            val_dataloaders.append(DataLoader(
                val_ds,
                batch_size=self.config.batch_size,
                shuffle=False
            ))
        
        model_idx = 0
        model_info = []
        
        for task_group in compressed_models:
            for model_data in task_group['models']:
                model = model_data['model']
                model_name = model_data['name']
                model_info.append(model_data)
                
                model.eval()
                print(f"\n   Evaluating {model_name}...")
                
                # اطمینان از دستگاه بودن مدل
                if next(model.parameters()).device != self.device:
                    model.to(self.device)
                
                for task_id, dataloader in enumerate(val_dataloaders):
                    all_preds, all_labels = [], []
                    
                    with torch.no_grad():
                        for batch in dataloader:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels']
                            
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                            
                            preds = torch.argmax(logits, dim=-1).cpu().numpy()
                            all_preds.extend(preds)
                            all_labels.extend(labels.cpu().numpy())
                    
                    if len(all_labels) > 0:
                        accuracy = accuracy_score(all_labels, all_preds)
                        performance_matrix[model_idx, task_id] = accuracy
                        
                        if task_id == task_group['task_id']:
                            print(f"      Task {tasks[task_id]['name']}: {accuracy:.4f} (specialized)")
                        else:
                            print(f"      Task {tasks[task_id]['name']}: {accuracy:.4f}")
                
                model_idx += 1
        
        self.results['performance_matrix'] = performance_matrix
        self.results['model_info'] = model_info
        
        print(f"\n📈 Performance Matrix Summary:")
        print(f"   Shape: {performance_matrix.shape}")
        print(f"   Average diagonal (specialization): {performance_matrix.diag().mean():.4f}")
        print(f"   Average off-diagonal (generalization): {performance_matrix.mean():.4f}")
        
        return performance_matrix, model_info, val_dataloaders
    
    def create_smart_ensemble(self, model_info, performance_matrix, num_tasks):
        print(f"\n{'='*60}")
        print("🧠 CREATING SMART WEIGHTED ENSEMBLE")
        print(f"{'='*60}")
        
        models = [info['model'] for info in model_info]
        
        ensemble = SmartWeightedEnsemble(
            num_models=len(models),
            num_tasks=num_tasks
        )
        
        ensemble.performance_matrix = performance_matrix
        
        for i, info in enumerate(model_info):
            if 'stats' in info and 'importance_scores' in info['stats']:
                avg_importance = np.mean([s['importance'] for s in info['stats']['importance_scores']])
                ensemble.base_weights[i] = avg_importance
        
        ensemble.base_weights = ensemble.base_weights / ensemble.base_weights.sum()
        
        self.ensemble = ensemble
        self.ensemble.models = models
        self.ensemble.model_info = model_info
        
        print(f"✅ Created ensemble with {len(models)} models")
        print(f"   Base weights: {ensemble.base_weights.tolist()}")
        
        return ensemble
    
    def test_ensemble(self, ensemble, tasks, val_dataloaders, test_method: str = "hybrid"):
        print(f"\n{'='*60}")
        print(f"🧪 TESTING ENSEMBLE (Method: {test_method})")
        print(f"{'='*60}")
        
        ensemble_results = {}
        
        for task_id, (task, dataloader) in enumerate(zip(tasks, val_dataloaders)):
            print(f"\n   Testing on Task: {task['name']}")
            
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels']
                    
                    all_logits_list = []
                    for model in ensemble.models:
                        # اطمینان از دستگاه
                        if next(model.parameters()).device != self.device:
                            model.to(self.device)
                            
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        all_logits_list.append(logits)
                    
                    all_logits = torch.stack(all_logits_list, dim=0)
                    
                    combined_logits = ensemble.combine_logits(
                        task_id=task_id,
                        models=ensemble.models,
                        all_logits=all_logits,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        method=test_method
                    )
                    
                    preds = torch.argmax(combined_logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    
                    if batch_idx % 10 == 0:
                        print(f"      Batch {batch_idx+1}/{len(dataloader)}...", end='\r')
            
            if len(all_labels) > 0:
                accuracy = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='weighted')
                
                ensemble_results[task['name']] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'samples': len(all_labels)
                }
                
                print(f"      ✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        avg_accuracy = np.mean([r['accuracy'] for r in ensemble_results.values()])
        avg_f1 = np.mean([r['f1'] for r in ensemble_results.values()])
        
        print(f"\n📊 Ensemble Summary:")
        print(f"   Average Accuracy: {avg_accuracy:.4f}")
        print(f"   Average F1: {avg_f1:.4f}")
        
        return ensemble_results, avg_accuracy, avg_f1
    
    def distill_ensemble(self, ensemble, tasks, val_dataloaders):
        print(f"\n{'='*60}")
        print("🔥 DISTILLING ENSEMBLE KNOWLEDGE")
        print(f"{'='*60}")
        
        student_name = self.config.base_models[0]
        print(f"   Creating student model: {student_name}")
        
        student = AutoModelForSequenceClassification.from_pretrained(
            student_name,
            num_labels=self.config.num_labels,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        student_compressed, _ = self.compressor.compress_model(student, "student")
        
        self.distiller = KnowledgeDistiller(
            teacher_ensemble=ensemble,
            temperature=self.config.temperature,
            alpha=0.7
        )
        
        student_distilled, losses = self.distiller.distill(
            student_compressed,
            val_dataloaders,
            num_epochs=3,
            lr=1e-4
        )
        
        print(f"✅ Distillation complete. Final loss: {losses[-1]:.4f}")
        
        return student_distilled
    
    def hyperparameter_optimization(self, tasks, val_dataloaders):
        print(f"\n{'='*60}")
        print("⚙️  HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        
        def objective(trial):
            energy_threshold = trial.suggest_float('energy_threshold', 0.7, 0.95, step=0.05)
            temperature = trial.suggest_float('temperature', 1.0, 4.0, step=0.5)
            ensemble_method = trial.suggest_categorical('ensemble_method', 
                                                      ['weighted', 'task_aware_attention', 'hybrid'])
            
            temp_compressor = AdaptiveSVDCompressor(energy_threshold=energy_threshold, device=str(self.device))
            
            try:
                model_name = self.config.base_models[0]
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=self.config.num_labels
                ).to(self.device)
                
                compressed, _ = temp_compressor.compress_model(model, "temp")
                compressed.eval()
                
                accuracies = []
                for dataloader in val_dataloaders[:2]:
                    all_preds, all_labels = [], []
                    
                    for batch in dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels']
                        
                        outputs = compressed(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_labels.extend(labels.cpu().numpy())
                    
                    if all_labels:
                        acc = accuracy_score(all_labels, all_preds)
                        accuracies.append(acc)
                
                score = np.mean(accuracies) if accuracies else 0
                
                del model, compressed
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                return score
                
            except Exception as e:
                print(f"   Trial error: {e}")
                return 0
        
        if self.config.enable_hyperparam_opt:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)
            
            print(f"\n✅ Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"   {key}: {value}")
            print(f"   Best score: {study.best_value:.4f}")
            
            self.config.energy_threshold = study.best_params['energy_threshold']
            self.config.temperature = study.best_params['temperature']
            self.config.ensemble_method = study.best_params['ensemble_method']
            
            return study.best_params
        else:
            print("   Skipped (disabled in config)")
            return {}
    
    def save_results(self, ensemble_results, distilled_model=None):
        print(f"\n{'='*60}")
        print("💾 SAVING RESULTS")
        print(f"{'='*60}")
        
        results_df = pd.DataFrame([
            {
                'task': task_name,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'samples': metrics['samples']
            }
            for task_name, metrics in ensemble_results.items()
        ])
        
        results_path = f"{self.config.output_dir}/ensemble_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to: {results_path}")
        
        if self.results['performance_matrix'] is not None:
            perf_df = pd.DataFrame(
                self.results['performance_matrix'].cpu().numpy(),
                columns=[t['name'] for t in self.results.get('tasks', [])],
                index=[info['name'] for info in self.results.get('model_info', [])]
            )
            perf_df.to_csv(f"{self.config.output_dir}/performance_matrix.csv")
            print(f"✅ Performance matrix saved")
        
        if self.results['compression_stats']:
            comp_df = pd.DataFrame(self.results['compression_stats'])
            comp_df.to_csv(f"{self.config.output_dir}/compression_stats.csv", index=False)
            print(f"✅ Compression stats saved")
        
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('__') and not callable(v)}
        
        with open(f"{self.config.output_dir}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Config saved")
        
        if distilled_model is not None:
            model_save_path = f"{self.config.output_dir}/distilled_model"
            torch.save({
                'model_state_dict': distilled_model.state_dict(),
                'config': config_dict
            }, f"{model_save_path}.pth")
            print(f"✅ Distilled model saved to: {model_save_path}.pth")
        
        self.create_summary_report(ensemble_results)
        
        print(f"\n🎉 All results saved in: {self.config.output_dir}")
    
    def create_summary_report(self, ensemble_results):
        report_path = f"{self.config.output_dir}/summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ADVANCED SVD ENSEMBLE - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 TASK PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            
            for task_name, metrics in ensemble_results.items():
                f.write(f"{task_name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  Samples: {metrics['samples']}\n\n")
            
            avg_acc = np.mean([m['accuracy'] for m in ensemble_results.values()])
            avg_f1 = np.mean([m['f1'] for m in ensemble_results.values()])
            
            f.write(f"\n📈 OVERALL AVERAGE:\n")
            f.write(f"  Accuracy: {avg_acc:.4f}\n")
            f.write(f"  F1 Score: {avg_f1:.4f}\n\n")
            
            if self.results['performance_matrix'] is not None:
                perf_matrix = self.results['performance_matrix']
                f.write("🎯 SPECIALIZATION VS GENERALIZATION:\n")
                f.write(f"  Avg. Diagonal (Specialization): {perf_matrix.diag().mean():.4f}\n")
                f.write(f"  Avg. Overall (Generalization): {perf_matrix.mean():.4f}\n\n")
            
            if self.results['compression_stats']:
                avg_ratio = np.mean([s.get('avg_compression_ratio', 0) 
                                   for s in self.results['compression_stats'] 
                                   if 'avg_compression_ratio' in s])
                f.write(f"🗜️  COMPRESSION STATS:\n")
                f.write(f"  Avg. Compression Ratio: {avg_ratio:.2%}\n")
                f.write(f"  Total Compressed Layers: {len(self.results['compression_stats'])}\n")
        
        print(f"✅ Summary report saved: {report_path}")
    
    def run_full_pipeline(self):
        print("\n" + "="*80)
        print("🚀 ADVANCED SVD ENSEMBLE PIPELINE - STARTING")
        print("="*80)

        print("\n🔍 STEP 0: Verifying Model and Tokenizer Connection...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_models[0],
                cache_dir=self.config.cache_dir,
                local_files_only=False
            )
            
            print(f"✅ Tokenizer loaded successfully from: {self.config.base_models[0]}")
            
            del tokenizer
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not load tokenizer.")
            print(f"   Error: {e}")
            print("   💡 Please check your internet connection or clear HuggingFace cache.")
            return

        print("🔍 STEP 1: Loading tasks... Done.")

        
        tasks = load_all_tasks(self.config.data_dir)
        if not tasks:
            print("❌ No valid tasks found!")
            return
        
        self.results['tasks'] = tasks
        
        print("🔍 STEP 2: Creating Tokenizers and DataLoaders...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_models[0])
        val_dataloaders = []
        
        for i, task in enumerate(tasks):
            print(f"   ⏳ Processing Task {i+1}/{len(tasks)}: {task['name']} ...")
            
            try:
                val_ds = MultiTaskDataset(
                    task['val_df']['text'],
                    task['val_df']['label'],
                    tokenizer,
                    max_length=self.config.max_length
                )
                
                val_dataloader = DataLoader(
                    val_ds,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
                
                val_dataloaders.append(val_dataloader)
                print(f"      ✅ Task {task['name']} loaded ({len(val_ds)} samples)")
                
            except Exception as e:
                print(f"      ❌ Error loading {task['name']}: {e}")
                continue
        
        print(f"✅ Total DataLoaders created: {len(val_dataloaders)}")
        
        best_params = self.hyperparameter_optimization(tasks, val_dataloaders)        
        print("🔍 STEP 3: ...")

        compressed_models = self.compress_all_models(tasks)
        print("\n🔍 STEP 3.5: FINE-TUNING COMPRESSED MODELS")
        print("="*60)

        # اصلاح ساخت DataLoader برای رفع مشکل مصرف منابع
        print("🔧 Optimizing DataLoaders for Performance...")
        
        # لیست مدل‌ها برای ساخت دیتاست
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_models[0])
        
        # ساخت Train Loaders برای فاین‌تیونینگ
        for task_group in compressed_models:
            task_name = task_group['task']['name']
            
            train_ds = MultiTaskDataset(
                task_group['task']['train_df']['text'],
                task_group['task']['train_df']['label'],
                tokenizer,
                max_length=self.config.max_length
            )
            
            # 👇 تغییرات کلیدی اینجا اعمال شد:
            train_loader = DataLoader(
                train_ds, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
                        
            val_ds = MultiTaskDataset(
                task_group['task']['val_df']['text'],
                task_group['task']['val_df']['label'],
                tokenizer,
                max_length=self.config.max_length
            )
            
            val_loader = DataLoader(
                val_ds, 
                batch_size=self.config.batch_size, 
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            for model_data in task_group['models']:
                model = model_data['model']
                model_name = model_data['name']
                
                print(f"\n   🎓 Starting Fine-tuning {model_name} on {task_name}...")
                print(f"   Device: {self.device}, Batch Size: {self.config.batch_size}")
                
                # فراخوانی تابع آموزش با لاگ‌های بهتر
                tuned_model, best_acc = fine_tune_compressed_model(
                    model=model,
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    device=self.device,
                    epochs=3,
                    lr=2e-5
                )
                
                print(f"   ✅ Best Validation Accuracy for {model_name}: {best_acc:.4f}")
                
                model_data['model'] = tuned_model
                del train_loader, val_loader, train_ds, val_ds # پاکسازی حافظه بین تسک‌ها
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("✅ Fine-tuning completed successfully!")
        print("🔍 STEP 4: CROSS-TASK EVALUATION")

        performance_matrix, model_info, val_dataloaders = self.evaluate_cross_task(
            compressed_models, tasks
        )
        
        print("🔍 STEP 5: ...")

        ensemble = self.create_smart_ensemble(
            model_info, performance_matrix, len(tasks)
        )
        
        print(f"\n{'='*60}")
        print("🔬 COMPARING ENSEMBLE METHODS")
        print(f"{'='*60}")
        
        methods = ['simple_average', 'weighted', 'task_aware_attention', 'hybrid']
        method_results = {}
        
        for method in methods:
            print(f"\nTesting method: {method}")
            results, avg_acc, avg_f1 = self.test_ensemble(
                ensemble, tasks, val_dataloaders, test_method=method
            )
            method_results[method] = {
                'results': results,
                'avg_accuracy': avg_acc,
                'avg_f1': avg_f1
            }
        
        best_method = max(method_results.items(), 
                         key=lambda x: x[1]['avg_accuracy'])[0]
        
        print(f"\n🏆 BEST METHOD: {best_method}")
        print(f"   Accuracy: {method_results[best_method]['avg_accuracy']:.4f}")
        print(f"   F1: {method_results[best_method]['avg_f1']:.4f}")
        
        distill_choice = input("\n🤔 Do you want to distill ensemble to a single model? (y/n): ")
        
        distilled_model = None
        if distill_choice.lower() == 'y':
            distilled_model = self.distill_ensemble(ensemble, tasks, val_dataloaders)
            
            print(f"\n📊 Testing distilled model...")
            distilled_model.eval()
            
            for task_id, (task, dataloader) in enumerate(zip(tasks, val_dataloaders)):
                all_preds, all_labels = [], []
                
                with torch.no_grad():
                    for batch in dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels']
                        
                        outputs = distilled_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_labels.extend(labels.cpu().numpy())
                
                if all_labels:
                    acc = accuracy_score(all_labels, all_preds)
                    f1 = f1_score(all_labels, all_preds, average='weighted')
                    print(f"   Task {task['name']}: Acc={acc:.4f}, F1={f1:.4f}")
        
        self.save_results(method_results[best_method]['results'], distilled_model)
        
        print(f"\n" + "="*80)
        print("🎉 PIPELINE COMPLETE - KEY FINDINGS")
        print("="*80)
        
        print(f"\n📊 ENSEMBLE PERFORMANCE:")
        print(f"   Best Method: {best_method}")
        print(f"   Average Accuracy: {method_results[best_method]['avg_accuracy']:.4f}")
        print(f"   Average F1: {method_results[best_method]['avg_f1']:.4f}")
        
        if performance_matrix is not None:
            print(f"\n🎯 KNOWLEDGE TRANSFER:")
            print(f"   Specialization (diagonal): {performance_matrix.diag().mean():.4f}")
            print(f"   Generalization (all): {performance_matrix.mean():.4f}")
            print(f"   Transfer Ratio: {performance_matrix.mean()/performance_matrix.diag().mean():.2%}")
        
        if self.results['compression_stats']:
            avg_ratio = np.mean([s.get('avg_compression_ratio', 0) 
                               for s in self.results['compression_stats'] 
                               if 'avg_compression_ratio' in s])
            print(f"\n🗜️  COMPRESSION EFFICIENCY:")
            print(f"   Average Compression Ratio: {avg_ratio:.2%}")
            print(f"   Memory Saved: {(1-avg_ratio):.2%}")
        
        print(f"\n💾 Results saved in: {self.config.output_dir}")
        print("="*80)

# ==================== 8. EXECUTION ====================
if __name__ == "__main__":
    try:
        assembler = AdvancedSVDAssembler(config)
        
        assembler.run_full_pipeline()
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        traceback.print_exc()