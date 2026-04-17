# ==================== main.py ====================

import os
import json
from pathlib import Path
from datetime import datetime
import time
import torch

# Import your modules
from bert_loader import load_original_model, save_model
from svd_compressor import compress_bert
from evaluator import run_evaluation
from config import config


def calculate_size_reduction(model_before, model_after):
    """Calculates size reduction in bytes and percentage."""
    try:
        size_before = sum(p.element_size() * p.nelement() for p in model_before.parameters())
        size_after = sum(p.element_size() * p.nelement() for p in model_after.parameters())
        
        reduction_bytes = size_before - size_after
        reduction_mb = reduction_bytes / (1024 * 1024)
        reduction_pct = (reduction_bytes / size_before) * 100 if size_before > 0 else 0
        
        return {
            "original_bytes": int(size_before),
            "compressed_bytes": int(size_after),
            "reduction_mb": round(reduction_mb, 2),
            "reduction_percent": round(reduction_pct, 4)
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("🚀 High-Precision BERT SVD Evaluation Pipeline")
    print("=" * 70)
    
    # Setup directories
    project_root = Path(__file__).resolve().parent.parent
    experiments_dir = project_root / "experiments"
    results_dir = experiments_dir / "results"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"results_{timestamp}.json"
    history_file = results_dir / "history.json"
    
    print(f"📂 Results will be saved to: {results_dir}")
    print(f"🎯 Target Rank: {config.svd_rank}")
    print(f"🖥️ Device: {config.device}")
    print("-" * 70)
    
    total_start_time = time.time()
    
    results_data = {
        "pipeline_id": timestamp,
        "config": {
            "model_name": config.model_name,
            "dataset": f"{config.dataset_name}/{config.dataset_subset}",
            "svd_rank": config.svd_rank,
            "device": str(config.device),
            "batch_size": config.batch_size
        },
        "original_model": {"metrics": {}, "stats": {}},
        "compressed_model": {"metrics": {}, "stats": {}},
        "comparison": {},
        "timestamp": datetime.now().isoformat(),
        "errors": []
    }
    
    try:
        # Step 1: Load and Evaluate Original Model
        print("\n[Step 1/4] Loading Original Model & Baseline Evaluation...")
        
        orig_model = None
        tokenizer = None
        orig_metrics = {"status": "pending", "accuracy": None}
        orig_stats = {}
        
        try:
            orig_model, tokenizer = load_original_model()
            
            if orig_model is None or tokenizer is None:
                raise ValueError("Model or Tokenizer returned None during loading.")
            
            orig_metrics = run_evaluation(orig_model, tokenizer)
            
            if isinstance(orig_metrics, dict) and "error" in orig_metrics:
                orig_metrics["status"] = "failed"
                results_data["errors"].append(f"Original Eval Error: {orig_metrics['error']}")
            else:
                orig_metrics["status"] = "success"
            
            orig_stats = calculate_size_reduction(orig_model, orig_model)
            
            print(f"✅ Original Model Loaded & Evaluated.")
            if orig_metrics.get("accuracy") is not None:
                print(f"   Accuracy: {orig_metrics['accuracy']:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in Step 1: {error_msg}")
            orig_metrics = {"status": "failed", "error": error_msg, "accuracy": None}
            orig_stats = {"error": error_msg}
            results_data["errors"].append(f"Step 1 Error: {error_msg}")
        
        print("\n[Step 2/4] Performing Input SVD Compression...")

        svd_model = None
        svd_metrics = {"status": "pending", "accuracy": None}
        svd_stats = {}

        try:
            # دریافت مدل فشرده شده
            svd_model, _ = compress_bert()
            
            if svd_model is None:
                raise RuntimeError("Input SVD compression returned None.")
            
            print("✅ Input SVD Compression completed successfully.")
            
            # محاسبه اندازه مدل (همان مدل اصلی است)
            if orig_model is not None:
                svd_stats = calculate_size_reduction(orig_model, svd_model)
            else:
                svd_stats = {"error": "Original model not available for comparison"}
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in Step 2: {error_msg}")
            svd_model = None
            svd_metrics = {"status": "failed", "error": error_msg, "accuracy": None}
            svd_stats = {"error": error_msg}
            results_data["errors"].append(f"Step 2 Error: {error_msg}")
                
        # Step 3: ارزیابی مدل فشرده شده
        print("\n[Step 3/4] Evaluating Compressed Model...")
        if svd_model is not None:
            svd_metrics = run_evaluation(
                model=svd_model,
                tokenizer=tokenizer,  # استفاده از توکنایزر اصلی
                save_results=True,
                experiment_type="svd_compressed"
            )
            print(f"✅ Compressed model evaluation completed. Accuracy: {svd_metrics['accuracy']:.4f}")
        else:
            print("⚠️ Skipping compressed model evaluation due to compression failure.")
            svd_metrics = {"status": "skipped", "accuracy": None}
        
        # Step 4: Save Results
        print("\n[Step 4/4] Saving Results...")
        
        # Store results
        results_data["original_model"]["metrics"] = orig_metrics
        results_data["original_model"]["stats"] = orig_stats
        results_data["compressed_model"]["metrics"] = svd_metrics
        results_data["compressed_model"]["stats"] = svd_stats
        
        # Calculate comparison
        acc_orig = orig_metrics.get('accuracy')
        acc_svd = svd_metrics.get('accuracy')
        
        if acc_orig is not None and acc_svd is not None:
            diff = acc_svd - acc_orig
            results_data["comparison"] = {
                "accuracy_loss": abs(diff),
                "accuracy_delta": diff,
                "size_reduction_mb": svd_stats.get('reduction_mb', 0),
                "size_reduction_pct": svd_stats.get('reduction_percent', 0)
            }
            
            print(f"\n📊 Performance Summary:")
            print(f"   Original Accuracy: {acc_orig:.4f}")
            print(f"   Compressed Accuracy: {acc_svd:.4f}")
            print(f"   Accuracy Change: {diff:+.4f}")
            print(f"   Size Reduction: {svd_stats.get('reduction_mb', 0):.2f} MB ({svd_stats.get('reduction_percent', 0):.2f}%)")
        else:
            results_data["comparison"] = {"error": "Could not compare due to missing accuracy values"}
        
        # Save individual result
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Individual Result Saved: {results_file.name}")
        except Exception as e:
            print(f"❌ Failed to save individual result: {e}")
            results_data["errors"].append(f"Save Error: {e}")
        
        # Update history
        try:
            history = []
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    try:
                        history = json.load(f)
                    except json.JSONDecodeError:
                        history = []
            
            history.append(results_data)
            
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            print(f"✅ History Updated: {history_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to update history: {e}")
        
        print("\n" + "=" * 70)
        print("🎉 Pipeline Execution Finished!")
        print("=" * 70)
        print(f"Results Location: {results_file.absolute()}")
        
    except KeyboardInterrupt:
        print("\n⛔ Process Interrupted by User.")
    except Exception as e:
        print(f"\n💥 Critical System Error: {e}")
        raise


if __name__ == "__main__":
    main()