# ==================== inspect_datasets.py ====================
"""
🔍 DATASET STRUCTURE INSPECTOR
Displays column names and sample data for all reasoning datasets.
Author: RAINO
"""

import pandas as pd
from pathlib import Path

def inspect_dataset(file_path, task_name):
    """Reads a CSV file and prints its structure and a sample row."""
    print(f"\n{'='*80}")
    print(f"📁 File: {file_path}")
    print(f"🏷️ Task: {task_name}")
    print(f"{'='*80}")
    
    try:
        # Read the file
        df = pd.read_csv(file_path)
        
        # Basic Info
        print(f"   📊 Total Rows: {len(df)}")
        print(f"   📊 Columns ({len(df.columns)}):")
        
        # List columns
        for i, col in enumerate(df.columns):
            print(f"      [{i}] {col}")
            
        print("\n   📋 Sample Data (First Row):")
        print("-" * 40)
        
        # Display first row values
        first_row = df.iloc[0]
        for col in df.columns:
            value = first_row[col]
            # Truncate long strings for readability
            if isinstance(value, str) and len(value) > 100:
                value = value[:97] + "..."
            elif isinstance(value, dict):
                value = str(value)[:50] + "..."
                
            print(f"   {col}: {value}")
            
        print("-" * 40)
        
        # Check specific types
        print("\n   🔬 Data Types:")
        for col in df.columns:
            dtype = df[col].dtype
            if 'object' in str(dtype):
                # Check if it looks like a list/dict string
                sample_val = df[col].iloc[0]
                if isinstance(sample_val, str):
                    if sample_val.startswith('[') or sample_val.startswith('{'):
                        print(f"      {col}: String (Likely JSON/List/Dict)")
                    else:
                        print(f"      {col}: String (Text)")
                else:
                    print(f"      {col}: {type(sample_val).__name__}")
            else:
                print(f"      {col}: {dtype}")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

def main():
    print("🚀 Starting Dataset Inspection...")
    
    # Define paths based on previous scan
    datasets = [
        ("commonsenseqa", "./data/reasoning/commonsenseqa/processed/commonsenseqa_train.csv"),
        ("hellaswag", "./data/reasoning/hellaswag/processed/hellaswag_train.csv"),
        ("gsm8k", "./data/reasoning/gsm8k/processed/gsm8k_train.csv"),
        ("arc_challenge", "./data/reasoning/arc_challenge/processed/arc_challenge_train.csv"),
        ("arc_agi", "./data/reasoning/arc_agi/processed/arc_agi_train.csv"),
    ]
    
    results = {}
    
    for task_name, file_path in datasets:
        success = inspect_dataset(file_path, task_name)
        results[task_name] = success
        
    print("\n" + "="*80)
    print("✅ Inspection Complete.")
    print("="*80)
    
    # Summary
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"⚠️  Failed to read: {failed}")
    else:
        print("🎉 All datasets inspected successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")