# model_versioning/hf_publisher.py
import os
import tempfile
from huggingface_hub import HfApi, Repository, create_repo, upload_folder
from django.conf import settings
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

class HuggingFacePublisher:
    def __init__(self, model_version, token=settings.HUGGINGFACE_TOKEN):
        self.version = model_version
        self.token = token
        self.api = HfApi()
        self.repo_name = f"{settings.HUGGINGFACE_USERNAME}/{self._generate_repo_name()}"
    
    def _generate_repo_name(self):
        run = self.version.run
        return f"{run.model_name.split('/')[-1]}-{run.task_name}-v{self.version.version_number}"
    
    def prepare_model_for_hub(self, output_dir):
        """Convert saved model to Hugging Face format."""
        # Load model from artifact path
        model = AutoModelForSequenceClassification.from_pretrained(
            self.version.artifact_path
        )
        
        # Add metadata from config version
        config = AutoConfig.from_pretrained(self.version.artifact_path)
        config.task_specific_params = {
            "task": self.version.run.task_name,
            "attention_type": self.version.run.attention_type,
            "training_method": self.version.run.method
        }
        
        # Save in HF format
        model.save_pretrained(output_dir)
        
        # Generate model card with metrics
        self._generate_model_card(output_dir)
        
        return output_dir
    
    def _generate_model_card(self, output_dir):
        """Create README.md with model details."""
        run = self.version.run
        metrics = self.version.metrics
        
        card = f"""---
language: en
tags:
- {run.task_name}
- {run.attention_type}
- transformer
metrics:
- name: accuracy
  value: {metrics.get('eval_accuracy', 'N/A')}
- name: f1
  value: {metrics.get('eval_f1', 'N/A')}
- name: loss
  value: {metrics.get('eval_loss', 'N/A')}
---

# {run.model_name} - {run.task_name.upper()}

This model was trained using the Transformer Pro Toolkit.

## Training Configuration
- **Base Model**: {run.model_name}
- **Attention Type**: {run.attention_type}
- **Training Method**: {run.method}
- **Epochs**: {run.epochs}
- **Batch Size**: {run.batch_size}
- **Learning Rate**: {run.learning_rate}
- **Max Sequence Length**: {run.max_seq_length}

## Performance
- **Accuracy**: {metrics.get('eval_accuracy', 'N/A')}
- **F1 Score**: {metrics.get('eval_f1', 'N/A')}
- **Loss**: {metrics.get('eval_loss', 'N/A')}

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("{self.repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("{self.repo_name}")
"""
with open(os.path.join(output_dir, "README.md"), "w") as f:
f.write(card)

def publish(self):
"""Upload model to Hugging Face Hub."""

