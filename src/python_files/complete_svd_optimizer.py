# ==================== complete_svd_optimizer.py (FIXED VERSION) ====================
"""
🌍 ULTIMATE SVD OPTIMIZER & BENCHMARK SUITE (ROBUST FIX)
================================================================
Fixes:
1. Removed invalid access to module.name.
2. Implemented safe module replacement logic.
3. Added strict=False for loading state_dict to handle architecture mismatches.
4. Improved architecture pattern matching.
================================================================
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# Import Config and SVD Layer
try:
    from config import config
    from svd_compressor import SVDLinearLayer
except ImportError as e:
    raise ImportError(f"Missing dependencies: {e}. Please ensure config.py and svd_compressor.py are present.")

# =============================================================================
# 1. CONFIGURATION & PATTERNS DATABASE
# =============================================================================

# تعریف الگوهای نام لایه برای هر معماری (بر اساس ساختار واقعی Transformers)
ARCH_PATTERNS = {
    'bert': [
        'attention.self.query', 
        'attention.self.key', 
        'attention.self.value', 
        'attention.output.dense'
    ],
    'distilbert': [
        'attention.q_lin', 
        'attention.k_lin', 
        'attention.v_lin', 
        'attention.out_lin'
    ],
    'roberta': [
        'attention.self.query', 
        'attention.self.key', 
        'attention.self.value', 
        'attention.output.dense'
    ],
    'xlnet': [
        'layer.layer.0.attention.hybrid_attn.q_linear', 
        'layer.layer.0.attention.hybrid_attn.k_linear', 
        'layer.layer.0.attention.hybrid_attn.v_linear',
        'layer.layer.0.attention.output.dense'
    ],
    'albert': [
        'attention_query', 
        'attention_key', 
        'attention_value', 
        'attention_output'
    ]
}

MODEL_NAMES_TO_TEST = [
    "distilbert-base-uncased-finetuned-sst-2-english", 
    "google-bert/bert-base-uncased",                   
    "roberta-base",                                  
]

RANKS_TO_TEST = [4, 8, 16, 32, 64, 128] 

DATASETS_CONFIG = [
    # {"name": "Custom Sentiment", "path": "./data/combined_master_dataset.csv"},
    # 01_irony.csv
    {"name": "o", "path": "./data/01_irony.csv"},
    {"name": "1", "path": "./data/02_negation.csv"},
    {"name": "2", "path": "./data/03_comparative.csv"},



    # Add more datasets here if available:
    # {"name": "Irony Test", "path": "./data/task_a_irony.csv"},
]

# =============================================================================
# 2. DATASET HANDLING
# =============================================================================

class GenericTextDataset(Dataset):
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

def load_dataset_from_csv(path):
    """Loads CSV and standardizes column names."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Find text column
    text_col = None
    for col in ['text', 'Text', 'TEXT', 'sentence', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    # Find label column
    label_col = None
    for col in ['label', 'Label', 'target', 'class', 'sentiment']:
        if col in df.columns:
            label_col = col
            break
            
    if not text_col or not label_col:
        raise ValueError(f"Could not find text/label columns in {path}. Available: {list(df.columns)}")
    
    print(f"   ✅ Loaded '{path}' ({len(df)} samples)")
    return df.rename(columns={text_col: 'text', label_col: 'label'})

def get_available_datasets():
    available = []
    for ds in DATASETS_CONFIG:
        if 'path' in ds and Path(ds['path']).exists():
            available.append({"name": ds['name'], "type": "csv", "path": ds['path']})
        elif 'hf_name' in ds:
            available.append({"name": ds['name'], "type": "hf", "config": ds})
    return available

# =============================================================================
# 3. CORE LOGIC: ARCHITECTURE DETECTION & COMPRESSION
# =============================================================================

def detect_architecture(model_name):
    name_lower = model_name.lower()
    if 'distilbert' in name_lower: return 'distilbert'
    if 'roberta' in name_lower: return 'roberta'
    if 'xlnet' in name_lower: return 'xlnet'
    if 'albert' in name_lower: return 'albert'
    if 'bert' in name_lower: return 'bert'
    return 'bert'

def get_layer_patterns(arch):
    return ARCH_PATTERNS.get(arch, ARCH_PATTERNS['bert'])

def replace_module_by_name(module, name, new_module):
    """
    Helper function to safely replace a module by its full path name.
    This avoids the .name attribute error and ensures correct hierarchy traversal.
    """
    parts = name.split('.')
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    child_name = parts[-1]
    setattr(parent, child_name, new_module)

def apply_svd_compression(model, rank, layer_patterns):
    """
    Applies SVD compression ONLY to attention projection layers.
    Returns the number of layers compressed.
    """
    compressed_count = 0
    total_linear_layers = 0
    
    # Iterate using named_modules to get both name string and module object
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_linear_layers += 1
            
            # Check if this layer matches any pattern for the current architecture
            is_target = False
            for pattern in layer_patterns:
                if pattern in name:
                    is_target = True
                    break
            
            if is_target:
                try:
                    # Create compressed layer
                    new_layer = SVDLinearLayer(module, rank)
                    
                    # Replace in model using safe helper
                    replace_module_by_name(model, name, new_layer)
                    compressed_count += 1
                    
                except Exception as e:
                    print(f"      ⚠️ Compression failed for {name}: {e}")
                    continue
    
    print(f"      🔧 Total Linear Layers: {total_linear_layers}, Target Layers Compressed: {compressed_count}")
    return compressed_count

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# =============================================================================
# 4. EVALUATION ENGINE
# =============================================================================

def evaluate_model_on_dataset(model, tokenizer, dataset_df, batch_size=32):
    """Evaluates model on a single dataset."""
    try:
        dataset = GenericTextDataset(
            dataset_df['text'], 
            dataset_df['label'], 
            tokenizer, 
            max_length=config.max_seq_length
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds, all_labels = [], []
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(config.device)
                attn_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits
                
                if logits.shape[-1] == 1:
                    preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm.tolist(),
            'samples': len(y_true)
        }
        
    except Exception as e:
        print(f"   ❌ Evaluation Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_speed(model, tokenizer, test_texts, num_runs=5):
    """Measures inference speed."""
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            inputs = tokenizer(test_texts, return_tensors="pt", truncation=True, padding=True).to(config.device)
            _ = model(**inputs)
            end = time.time()
            times.append(end - start)
    return sum(times) / len(times)

# =============================================================================
# 5. MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_complete_optimization():
    print("="*80)
    print("🚀 ULTIMATE SVD OPTIMIZER: FULL SYSTEM TEST (FIXED)")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.device}")
    print("-"*80)
    
    results_log = []
    best_config = None
    best_accuracy = -1.0
    
    # Get Datasets
    datasets_list = get_available_datasets()
    if not datasets_list:
        print("❌ No valid datasets found. Please add paths to DATASETS_CONFIG.")
        return

    print(f"\n📂 Found {len(datasets_list)} datasets to test.\n")

    # Test Models
    for model_name in MODEL_NAMES_TO_TEST:
        print(f"\n{'='*80}")
        print(f"🔄 TESTING MODEL: {model_name}")
        print('='*80)
        
        try:
            # Load Tokenizer & Model
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
            
            # IMPORTANT: Use ignore_mismatched_sizes=True to avoid errors when loading 
            # a base model into a classification head that expects different output sizes
            model_base = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2, 
                cache_dir=str(config.hf_cache_dir),
                ignore_mismatched_sizes=True 
            )
            model_base.to(config.device)
            
            arch = detect_architecture(model_name)
            patterns = get_layer_patterns(arch)
            
            print(f"   Detected Architecture: {arch.upper()}")
            print(f"   Target Patterns: {patterns[:2]}...")
            
            # Count initial target layers
            initial_targets = 0
            for name, module in model_base.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(p in name for p in patterns):
                        initial_targets += 1
            print(f"   Initial Target Layers Found: {initial_targets}")

            # Loop through Ranks
            for rank in RANKS_TO_TEST:
                print(f"\n   >>> Testing Rank: {rank}")
                
                # Clone model to avoid modifying base
                # We create a new instance and load weights
                temp_model = type(model_base)(model_base.config).to(config.device)
                temp_model.load_state_dict(model_base.state_dict(), strict=False)
                
                # Compress
                layers_compacted = apply_svd_compression(temp_model, rank, patterns)
                
                if layers_compacted == 0:
                    print(f"      ⚠️ No target layers found for Rank {rank}. Skipping.")
                    continue
                
                # Calculate Params Reduction
                orig_params = count_parameters(model_base)
                comp_params = count_parameters(temp_model)
                reduction_pct = ((orig_params - comp_params) / orig_params) * 100
                
                # Evaluate on ALL Datasets
                dataset_results = {}
                avg_acc = 0.0
                valid_datasets = 0
                
                for ds in datasets_list:
                    if ds['type'] == 'csv':
                        try:
                            df = load_dataset_from_csv(ds['path'])
                            metrics = evaluate_model_on_dataset(temp_model, tokenizer, df)
                            
                            if metrics:
                                dataset_results[ds['name']] = metrics
                                avg_acc += metrics['accuracy']
                                valid_datasets += 1
                                
                                print(f"      [{ds['name']}] Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                        except Exception as e:
                            print(f"      ❌ Dataset loading error: {e}")
                            continue
                
                if valid_datasets == 0:
                    print(f"      ❌ No successful evaluations for Rank {rank}.")
                    continue
                
                avg_acc /= valid_datasets
                
                # Store Result
                result_entry = {
                    "model": model_name,
                    "architecture": arch,
                    "rank": rank,
                    "layers_compressed": layers_compacted,
                    "params_reduction_pct": round(reduction_pct, 2),
                    "avg_accuracy": round(avg_acc, 4),
                    "best_dataset_acc": max([r['accuracy'] for r in dataset_results.values()]),
                    "status": "Success"
                }
                results_log.append(result_entry)
                
                # Track Best
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                    best_config = result_entry.copy()
                
                print(f"      📊 Avg Accuracy: {avg_acc:.4f} | Params Reduced: {reduction_pct:.1f}%")

        except Exception as e:
            print(f"❌ Critical Error loading/testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # =============================================================================
    # 6. FINAL REPORTING
    # =============================================================================
    print("\n" + "="*80)
    print("🏆 FINAL OPTIMIZATION REPORT")
    print("="*80)
    
    if not results_log:
        print("No successful runs recorded.")
        return

    # Convert to DataFrame for pretty printing
    df_results = pd.DataFrame(results_log)
    
    # Sort by Accuracy descending
    df_results = df_results.sort_values(by='avg_accuracy', ascending=False)
    
    print("\n📋 COMPLETE RESULTS TABLE:")
    print(df_results[['model', 'architecture', 'rank', 'layers_compressed', 'params_reduction_pct', 'avg_accuracy']].to_string(index=False))
    
    if best_config:
        print("\n" + "🎉 BEST CONFIGURATION FOUND 🎉")
        print("-"*80)
        print(f"Model:       {best_config['model']}")
        print(f"Architecture:{best_config['architecture'].upper()}")
        print(f"Rank:        {best_config['rank']}")
        print(f"Layers Comp: {best_config['layers_compressed']}")
        print(f"Params Red.: {best_config['params_reduction_pct']}%")
        print(f"Avg Acc.:    {best_config['avg_accuracy']:.4f}")
        print("-"*80)
        print("💡 Recommendation: Use this configuration for deployment.")
        print("   It offers the best trade-off between speed/memory and accuracy.")
    
    # Save detailed report to JSON
    report_path = config.results_dir / f"svd_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "best_config": best_config,
                "all_results": results_log
            }, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️ Could not save report to JSON: {e}")
    
    print("="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_complete_optimization()