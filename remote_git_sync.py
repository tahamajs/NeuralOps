import subprocess, os, sys

os.chdir("D:/SVD_linear_Attention")

files_to_stage = [
    "neuralops/attentions/adaptive_svd_attention.py",
    "neuralops/attentions/aegis_hybrid.py",
    "neuralops/attentions/aegis_spectron_fp8.py",
    "neuralops/attentions/asvd_attention.py",
    "scripts/run_kd_all.py",
]

files_to_rm = [
    "results/kd_cache/gpt2_062dd3028ace/manifest.json",
    "results/kd_cache/gpt2_4992e55e035c/manifest.json",
    "results/kd_cache/gpt2_94646393f7bd/manifest.json",
]

files_to_clean = [
    "fix_compile.py", "fix_lora.py", "fix_lora2.py", "fix_modules.py",
    "fix_q_transpose.py", "fix_sdpa.py", "fix_triton_math.py",
    "fix_triton_math2.py", "fix_triton_round.py", "run_all_fixed.bat",
    "run_both.bat", "test_cproj.py", "test_div.py", "test_norms.py",
    "test_q.py", "test_ua.py",
]

for f in files_to_clean:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned: {f}")
    except Exception as e:
        print(f"Failed to clean {f}: {e}")

r = subprocess.run(["git", "add"] + files_to_stage, capture_output=True, text=True)
print("add tracked:", r.returncode, r.stderr.strip())

r = subprocess.run(["git", "rm", "--cached", "--ignore-unmatch"] + files_to_rm, capture_output=True, text=True)
print("rm cached:", r.returncode, r.stdout.strip(), r.stderr.strip())

r = subprocess.run(["git", "commit", "-m",
    "fix: sync server changes - lora_alpha scaling, triton purge, cache cleanup"],
    capture_output=True, text=True)
print("commit:", r.returncode, r.stdout.strip(), r.stderr.strip())

if r.returncode == 0:
    r2 = subprocess.run(["git", "push", "origin", "main", "--force-with-lease"],
        capture_output=True, text=True)
    print("push:", r2.returncode, r2.stdout.strip(), r2.stderr.strip())
elif "nothing to commit" in r.stdout:
    print("Nothing to commit; will merge then push")
    r2 = subprocess.run(["git", "pull", "--rebase", "origin", "main"],
        capture_output=True, text=True)
    print("rebase:", r2.returncode, r2.stdout.strip(), r2.stderr.strip())
    r3 = subprocess.run(["git", "push", "origin", "main"],
        capture_output=True, text=True)
    print("push after rebase:", r3.returncode, r3.stdout.strip(), r3.stderr.strip())
