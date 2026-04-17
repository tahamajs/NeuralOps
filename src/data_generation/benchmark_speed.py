# ==================== benchmark_speed.py ====================
import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from svd_compressor import compress_bert_true_svd
from config import config

def measure_inference_time(model, tokenizer, text_list, num_runs=5):
    """
    Measures the average inference time for a list of texts.
    """
    model.eval()
    with torch.no_grad():
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            # Tokenize and run
            inputs = tokenizer(text_list, return_tensors="pt", truncation=True, padding=True).to(config.device)
            outputs = model(**inputs)
            
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Clear cache to avoid GPU memory issues affecting speed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    print("="*60)
    print("BENCHMARKING: Original vs Compressed Model Speed")
    print("="*60)
    
    # 1. Load Original Model (Assuming it's already trained/fine-tuned)
    # If you haven't trained it yet, load the base one just for structure comparison
    print("\n🔄 Loading Original Model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=str(config.hf_cache_dir))
    
    # Note: For accurate speed comparison, use the TRAINED models.
    # If you don't have them saved yet, this will load the base model which is untrained.
    # Let's assume you have 'original_finetuned' and 'finetuned_svd_rank_32' directories.
    
    try:
        original_path = config.output_dir / "original_finetuned"
        compressed_path = config.output_dir / f"finetuned_svd_rank_{config.svd_rank}"
        
        model_orig = AutoModelForSequenceClassification.from_pretrained(original_path)
        model_comp = AutoModelForSequenceClassification.from_pretrained(compressed_path)
        
        model_orig.to(config.device)
        model_comp.to(config.device)
        
        print(f"✅ Loaded Trained Original Model from: {original_path}")
        print(f"✅ Loaded Trained Compressed Model from: {compressed_path}")
        
    except Exception as e:
        print(f"⚠️ Could not find trained models ({e}). Loading base models for structure test only.")
        model_orig = AutoModelForSequenceClassification.from_pretrained(config.model_name, cache_dir=str(config.hf_cache_dir))
        model_comp = compress_bert_true_svd(model_orig, config.svd_rank)
        model_orig.to(config.device)
        model_comp.to(config.device)
        print("⚠️ Warning: Comparing UNTRAINED models. Accuracy will be random, but SPEED should show improvement.")

    # Prepare test data
    test_texts = [
        "This movie was absolutely amazing and I loved every minute of it.",
        "The performance was terrible and the plot made no sense at all.",
        "A truly remarkable experience that exceeded all expectations."
    ] * 100 # Repeat to get stable timing
    
    print(f"\n🧪 Testing on {len(test_texts)} samples...")
    
    # Measure Original
    print("\n⏱️ Measuring Original Model Time...")
    t_orig = measure_inference_time(model_orig, tokenizer, test_texts)
    print(f"   Average Time: {t_orig:.4f} seconds")
    
    # Measure Compressed
    print("\n⏱️ Measuring Compressed Model Time...")
    t_comp = measure_inference_time(model_comp, tokenizer, test_texts)
    print(f"   Average Time: {t_comp:.4f} seconds")
    
    # Calculate Improvement
    improvement = ((t_orig - t_comp) / t_orig) * 100
    
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    print(f"Original Time:  {t_orig:.4f}s")
    print(f"Compressed Time: {t_comp:.4f}s")
    print(f"Speed Up:       {t_orig/t_comp:.2f}x faster")
    print(f"Time Saved:     {(t_orig - t_comp)*1000:.2f} ms per batch")
    print(f"Reduction %:    {improvement:.2f}%")
    print("="*60)
    
    if t_comp < t_orig:
        print("✅ SUCCESS! Compression reduced inference time.")
    else:
        print("⚠️ WARNING: Compressed model is slower (likely due to overhead or small dataset).")
        print("   Try increasing batch size or number of test samples.")

if __name__ == "__main__":
    main()