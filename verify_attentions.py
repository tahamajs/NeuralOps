import os
import importlib
import traceback
import sys

attention_dir = "Codes/framework/attentions"
sys.path.append(os.path.abspath("Codes"))

failed_modules = []
for file in os.listdir(attention_dir):
    if file.endswith(".py") and not file.startswith("__"):
        module_name = file[:-3]
        try:
            importlib.import_module(f"framework.attentions.{module_name}")
        except Exception as e:
            failed_modules.append((module_name, traceback.format_exc()))

if not failed_modules:
    print("All modules imported successfully without syntax/type errors!")
else:
    print(f"{len(failed_modules)} modules failed to import:")
    for mod, err in failed_modules:
        print(f"\n--- {mod} ---")
        print(err)
