import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

callback_code = """
import json
from transformers import TrainerCallback

class LiveHistoryCallback(TrainerCallback):
    def __init__(self, out_dir):
        self.history_file = Path(out_dir) / "history" / "log_history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.log_history:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(state.log_history, f, indent=2, default=str)
"""

if "LiveHistoryCallback" not in content:
    # Insert the callback class definition after imports
    content = content.replace("def run_train", callback_code + "\ndef run_train")
    
    # Inject it into the callbacks list
    content = content.replace(
        "callbacks = [fw[\"MemoryCallback\"](), fw[\"TimeCallback\"]()]",
        "callbacks = [fw[\"MemoryCallback\"](), fw[\"TimeCallback\"](), LiveHistoryCallback(out_dir)]"
    )

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
