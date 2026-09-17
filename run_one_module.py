import sys, os, json, traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"D:\SVD_Project\.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\SVD_Project\.hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\SVD_Project\.hf_cache"

sys.path.insert(0, "Codes")

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

MODEL_PATHS = {
    "distilbert": r"models\distilbert_sst2",
    "roberta": r"models\roberta-base",
    "gpt2": r"models\gpt2",
}

def test_one(module_name):
    result = {"module": module_name, "models": {}}
    mod = None
    try:
        mod = __import__(f"framework.attentions.{module_name}", fromlist=["*"])
    except ModuleNotFoundError:
        try:
            mod = __import__(f"framework.compression_engine.{module_name}", fromlist=["*"])
        except Exception as e:
            result["import_error"] = f"{type(e).__name__}: {e}"
            return result
    except Exception as e:
        result["import_error"] = f"{type(e).__name__}: {e}"
        return result


    inject_fn = None
    for name in dir(mod):
        if name.startswith("inject_"):
            inject_fn = getattr(mod, name)
            break

    if inject_fn is None:
        result["no_injector"] = True
        return result

    for model_name, model_path in MODEL_PATHS.items():
        try:
            cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
            if getattr(cfg, "pad_token_id", None) is None:
                cfg.pad_token_id = getattr(cfg, "eos_token_id", 0) or 0
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, config=cfg, local_files_only=True
            )
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = model.config.eos_token_id or 0
            model.eval()

            ids = torch.randint(0, 1000, (2, 16))
            attn = torch.ones_like(ids)
            with torch.no_grad():
                before = model(input_ids=ids, attention_mask=attn).logits.clone()
            modules_before = set(type(m).__name__ for m in model.modules())
            try:
                inject_fn(model=model, model_config=cfg)
            except TypeError:
                inject_fn(model=model)
            modules_after = set(type(m).__name__ for m in model.modules())
            actually_injected = (modules_before != modules_after)

            with torch.no_grad():
                after = model(input_ids=ids, attention_mask=attn).logits
            diff = (before - after).abs().max().item()
            status = "OK"
            if diff >= 100:
                status = "BAD"
            elif not actually_injected:
                status = "NO_OP_INJECTION"

            result["models"][model_name] = {
                "status": status,
                "diff": float(diff),
                "injected": actually_injected,
            }
        except Exception as e:
            result["models"][model_name] = {
                "status": "FAIL",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "traceback": traceback.format_exc(),
            }
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_one_module.py <module_name>")
        sys.exit(1)
    mod_name = sys.argv[1]
    res = test_one(mod_name)
    os.makedirs("audit", exist_ok=True)
    with open("audit/module_test_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=2))
