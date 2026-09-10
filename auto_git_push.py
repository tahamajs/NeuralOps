import time
import subprocess
import os

print("Starting Auto Git Push service...")
while True:
    try:
        # Add everything in results, except weights
        subprocess.run(["git", "add", "results/*.csv", "results/*.json", "results/*.png"], cwd=r"D:\SVD_Project")
        subprocess.run(["git", "add", "results/*/*.png", "results/*/*.csv", "results/*/*.json"], cwd=r"D:\SVD_Project")
        
        status = subprocess.run(["git", "status", "--porcelain"], cwd=r"D:\SVD_Project", capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", "chore(results): auto-sync latest GPU benchmark results"], cwd=r"D:\SVD_Project")
            subprocess.run(["git", "push", "origin", "main"], cwd=r"D:\SVD_Project")
            print("Pushed new results to GitHub.")
        else:
            print("No new results to push.")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(600)  # Sleep for 10 minutes
