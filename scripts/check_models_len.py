import os
import json

def check_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("models directory not found")
        return
    for d in sorted(os.listdir(models_dir)):
        p = os.path.join(models_dir, d, "config.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    max_pos = cfg.get("max_position_embeddings") or cfg.get("n_positions") or cfg.get("max_sequence_length")
                    model_type = cfg.get("model_type", "unknown")
                    print(f"{d:30s} | type: {model_type:15s} | max_len: {max_pos}")
            except Exception as e:
                print(f"{d:30s} | error: {e}")

if __name__ == "__main__":
    check_models()
