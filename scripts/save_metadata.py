import json, socket, subprocess, sys, time
from pathlib import Path
import torch

def get_git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd="D:/SVD_Project"
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def get_env():
    import transformers, datasets, numpy
    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": get_git_sha(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "numpy_version": numpy.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                       if torch.cuda.is_available() else None,
    }

def save_env_snapshot():
    env = get_env()
    Path("audit").mkdir(exist_ok=True)
    with open("audit/zeroshot_environment.json", "w") as f:
        json.dump(env, f, indent=2)
    return env

def save_run_metadata(output_dir, module, model, task, extra=None):
    meta = get_env()
    meta.update({"module": module, "model": model, "task": task, "extra": extra or {}})
    out = Path(output_dir) / "run_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    print(json.dumps(save_env_snapshot(), indent=2))
