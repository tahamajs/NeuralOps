# framework/quick.py
"""
Quick integration functions for easy use.
One-line functions to apply compression to any transformer model.
"""
from .soa import SOAConfig, SOABert, migrate_bert_to_soa, compute_global_basis_from_hidden_states

import torch
from typing import Optional, Union, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def quick_soa_model(model_name="bert-base-uncased", rank=64, num_labels=2, device="auto"):
    """
    Load a BERT model, compute basis, and return an SOA model.
    """
    from transformers import BertModel
    from .core.device import get_device

    device = get_device(device)
    bert = BertModel.from_pretrained(model_name).to(device)
    basis = compute_global_basis_from_hidden_states(bert, rank, sample_size=32)
    cfg = SOAConfig(k_rank=rank, num_labels=num_labels)
    soa = SOABert(cfg).to(device)
    soa = migrate_bert_to_soa(bert, soa, basis)
    return soa

def quick_svd_compress(
    model_name: str = "bert-base-uncased",
    rank: int = 64,
    attention_type: str = "svd",
    task: str = "sequence_classification",
    num_labels: int = 2,
    device: str = "auto"
) -> torch.nn.Module:
    """
    One-line function to load and compress a transformer model.
    
    Args:
        model_name: HuggingFace model name
        rank: SVD compression rank
        attention_type: Type of attention ('svd', 'nystrom', 'performer')
        task: Task type ('sequence_classification', 'qa', etc.)
        num_labels: Number of labels (for classification)
        device: Device to use
        
    Returns:
        Compressed model ready for use
    
    Example:
        >>> from framework.quick import quick_svd_compress
        >>> 
        >>> # Load and compress BERT
        >>> model = quick_svd_compress("bert-base-uncased", rank=64)
        >>> 
        >>> # Load and compress for MRPC
        >>> model = quick_svd_compress("bert-base-uncased", rank=32, num_labels=2)
    """
    from .core.model_wrapper import UnifiedTransformerModel
    from .core.device import get_device
    
    # Load model
    model = UnifiedTransformerModel.from_pretrained(
        model_name, task=task, num_labels=num_labels
    )
    
    # Replace attention
    model.replace_attention(attention_type, rank=rank)
    
    # Move to device
    device = get_device(device)
    model.to(device)
    
    print(f"✅ Loaded and compressed {model_name} (rank={rank}) on {device}")
    print(f"   Parameters: {model.get_parameter_count()}")
    
    return model


def quick_finetune(
    model: torch.nn.Module,
    train_dataset,
    eval_dataset,
    output_dir: str = "./output",
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5
):
    """
    One-line function to fine-tune a compressed model.
    
    Example:
        >>> from framework.quick import quick_finetune
        >>> 
        >>> model = quick_svd_compress("bert-base-uncased", rank=64)
        >>> quick_finetune(model, train_dataset, eval_dataset)
    """
    from .training import create_trainer
    
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    
    trainer.train()
    return trainer


def quick_inference(
    model: torch.nn.Module,
    text: Union[str, List[str]],
    tokenizer=None,
    max_length: int = 128
) -> torch.Tensor:
    """
    One-line inference with compressed model.
    
    Example:
        >>> from framework.quick import quick_inference
        >>> 
        >>> model = quick_svd_compress("bert-base-uncased", rank=64)
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> result = quick_inference(model, "This movie is great!", tokenizer)
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            getattr(model, 'config', {}).get('_name_or_path', 'bert-base-uncased')
        )
    
    if isinstance(text, str):
        text = [text]
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs