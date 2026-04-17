# ==================== config.py ====================
# این فایل کامل و اصلاح شده برای پیپ‌لاین SVD + Fine-Tuning است

import os
import torch
from pathlib import Path
from typing import Dict, Any
import copy
from datetime import datetime

BASE_DIR = Path.home() / "Documents" / "huggingface_secure_cache"
HF_CACHE_DIR = BASE_DIR / "models"
DATASET_CACHE_DIR = BASE_DIR / "datasets"
EXPERIMENTS_DIR = BASE_DIR / "experiments_logs"

for path in [BASE_DIR, HF_CACHE_DIR, DATASET_CACHE_DIR, EXPERIMENTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
os.environ["HF_HOME"] = str(BASE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(DATASET_CACHE_DIR)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[CONFIG] MPS available. Using Apple Silicon GPU.")
        # افزایش دقت ماتریس برای MPS
        torch.set_float32_matmul_precision('high')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[CONFIG] CUDA available. Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CONFIG] No GPU/MPS found. Falling back to CPU.")
    return device


DEVICE = get_device()

# --- پیکربندی دیتاست‌ها ---
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # "mrpc": {
    #     "name": "glue",
    #     "subset": "mrpc",
    #     "num_labels": 2,
    #     "max_length": 128,
    #     "text_columns": ["sentence1", "sentence2"],
    #     "label_column": "label",
    #     "local_path": None,
    # },
    "sst2": {
        "name": "glue",
        "subset": "sst2",
        "num_labels": 2,
        "max_length": 512,
        "text_columns": ["sentence"],
        "label_column": "label",
        "local_path": "./data/data_sst2.csv",
    },
    # "imdb": {
    #     "name": "imdb",
    #     "subset": None,
    #     "num_labels": 2,
    #     "max_length": 512,
    #     "text_columns": ["text"],
    #     "label_column": "label",
    #     "local_path": "./data/data_imdb.csv",
    # },
    "custom": {
        "name": "csv",
        "subset": None,
        "num_labels": 2,
        "max_length": 128,
        "text_columns": ["text"],
        "label_column": "label",
        # مسیر دیتاست شما
        "local_path": "./data/data_custom.csv", 
    }
}

CURRENT_DATASET_NAME = "custom"


def get_dataset_config(name: str) -> Dict[str, Any]:
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    return copy.deepcopy(DATASET_CONFIGS[name])


ACTIVE_DATASET_CONFIG = get_dataset_config(CURRENT_DATASET_NAME)


class Config:
    def __init__(self):
        self.model_name = "google-bert/bert-base-uncased"
        
        self.dataset_name = ACTIVE_DATASET_CONFIG['name']
        self.dataset_subset = ACTIVE_DATASET_CONFIG['subset']
        self.dataset_text_cols = ACTIVE_DATASET_CONFIG['text_columns']
        self.dataset_label_col = ACTIVE_DATASET_CONFIG['label_column']
        self.max_seq_length = ACTIVE_DATASET_CONFIG['max_length']
        self.num_labels = ACTIVE_DATASET_CONFIG['num_labels']
        self.local_dataset_path = ACTIVE_DATASET_CONFIG.get('local_path')
        
        self.svd_rank = 32  # رتبه SVD برای فشرده‌سازی ورودی
        self.enable_svd_projection = True
        
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.do_finetuning = True
        
        self.hf_cache_dir = HF_CACHE_DIR
        self.dataset_cache_dir = DATASET_CACHE_DIR
        
        self.output_dir = EXPERIMENTS_DIR / CURRENT_DATASET_NAME
        self.svd_model_dir = self.output_dir / "svd_compressed"

        self.results_dir = self.output_dir / "results"
        
        self.finetuned_model_dir = self.output_dir / "finetuned_svd_rank_{}_lr{}".format(
            self.svd_rank, 
            str(self.learning_rate).replace('.', '_')
        )
        
        self.device = DEVICE
        self.verbose = True
        
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  Model: {self.model_name}\n"
            f"  Dataset: {self.dataset_name}/{self.dataset_subset}\n"
            f"  Labels: {self.num_labels}\n"
            f"  Max Length: {self.max_seq_length}\n"
            f"  SVD Rank: {self.svd_rank}\n"
            f"  Fine-Tuning: {'Enabled' if self.do_finetuning else 'Disabled'}\n"
            f"  Epochs: {self.num_epochs}\n"
            f"  Device: {self.device}\n"
            f"  Pipeline ID: {self.pipeline_id}\n"
            f")"
        )


config = Config()